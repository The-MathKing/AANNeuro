"""Does the H3 signal survive adjustment for neuronal composition? (post hoc, added after the pre-registered analysis)
Neuronal score = mean within-sample rank percentile of the neuronal marker genes PRESENT IN BOTH cohorts
(RBFOX3, SNAP25, SYP, NEFL, SLC17A7, GAD1, MAP2; CAMK2A is absent from the lockbox deposit). Rank percentiles are
platform-agnostic, so the score can be computed in each cohort with no external reference.
 (a) lockbox AUC of the neuronal score alone;
 (b) logistic AD ~ frozen score + neuronal score + age + sex inside the lockbox (score OR per SD, LR test, AUC);
 (c) model re-trained on expression residualised for the neuronal score (OLS fitted on training samples only; the
     lockbox is residualised with the training coefficients) and evaluated on the lockbox."""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from src.load import load_gse118553, load_gse132903
from src.harmonize.cross_platform import compute_sample_ranks
from src.models.linear_pipeline import InFoldLinearModel
from src.eval.metrics import calc_auc
from src.eval.bootstrap import donor_bootstrap_ci
from src.eval.ledger_logger import log_analysis

NEURONAL = ["RBFOX3", "SNAP25", "SYP", "NEFL", "SLC17A7", "GAD1", "MAP2"]


def _ci(y, s):
    d = pd.DataFrame({"y_true": y, "y_pred": s, "donor_id": np.arange(len(y))})
    r = donor_bootstrap_ci(lambda x: calc_auc(x.y_true.values, x.y_pred.values), d, n_boot=2000)
    return float(r["estimate"]), float(r["ci_lower"]), float(r["ci_upper"])


def run_h3_composition():
    m1, e1 = load_gse118553(); m2, e2 = load_gse132903()
    tr = m1[(m1.region == "Temporal_Cortex") & m1.diagnosis.isin(["CTL", "AD"])]
    ytr = (tr.diagnosis == "AD").astype(int).values
    lb = m2[m2.diagnosis.isin(["CTL", "AD"])]
    y = (lb.diagnosis == "AD").astype(int).values
    markers = [g for g in NEURONAL if g in e1.index and g in e2.index]
    n_tr = compute_sample_ranks(e1[tr.index]).loc[markers].mean(0).values
    n_lb = compute_sample_ranks(e2[lb.index]).loc[markers].mean(0).values

    # frozen model (identical to H3)
    mod = InFoldLinearModel("ridge_classifier", top_hvg=1000, seed=42).fit_transform_train(e1[tr.index].T, ytr)
    p = np.clip(mod.predict(e2[lb.index].T), 1e-12, 1 - 1e-12)
    score = np.log(p / (1 - p))
    z = lambda v: (v - v.mean()) / v.std()
    out = {"markers_used": markers, "n_markers": len(markers)}
    out["corr_score_neuronal_lockbox"] = float(stats.spearmanr(score, n_lb)[0])
    out["neuronal_alone_auc_lockbox"] = calc_auc(y, -n_lb)  # AD expected lower
    out["neuronal_alone_ci"] = list(_ci(y, -n_lb)[1:])
    out["neuronal_alone_auc_train"] = calc_auc(ytr, -n_tr)
    age = lb.age.values; fem = (lb.sex == "FEMALE").astype(float).values
    Xf = sm.add_constant(np.c_[z(score), z(n_lb), age, fem]); Xr = sm.add_constant(np.c_[z(n_lb), age, fem])
    f, r = sm.Logit(y, Xf).fit(disp=0), sm.Logit(y, Xr).fit(disp=0)
    lr = 2 * (f.llf - r.llf)
    out["adj_or_per_sd"] = float(np.exp(f.params[1])); out["adj_or_ci"] = [float(v) for v in np.exp(f.conf_int()[1])]
    out["adj_lr_p"] = float(stats.chi2.sf(lr, 1))
    out["adj_auc_neuronal_age_sex"] = calc_auc(y, r.predict(Xr)); out["adj_auc_plus_score"] = calc_auc(y, f.predict(Xf))
    # (c) retrain with neuronal score regressed out of expression (training coefficients only)
    genes = [g for g in e1.index if g in e2.index]
    Xtr = e1[tr.index].loc[genes].T.values; Xlb = e2[lb.index].loc[genes].T.values
    A = sm.add_constant(n_tr); B = np.linalg.lstsq(A, Xtr, rcond=None)[0]
    Xtr_r = Xtr - A @ B; Xlb_r = Xlb - sm.add_constant(n_lb, has_constant="add") @ B
    m3 = InFoldLinearModel("ridge_classifier", top_hvg=1000, seed=42).fit_transform_train(pd.DataFrame(Xtr_r, columns=genes), ytr)
    s3 = m3.predict(pd.DataFrame(Xlb_r, columns=genes))
    est, lo, hi = _ci(y, s3)
    out["resid_neuronal_model_auc"] = est; out["resid_neuronal_model_ci"] = [lo, hi]
    json.dump(out, open("results/h3_composition.json", "w"), indent=2)
    log_analysis(analysis_id="H3_COMP_001", category="SENSITIVITY",
                 hypothesis_or_idea="H3 lockbox signal adjusted for neuronal marker score (post hoc)",
                 description="Neuronal rank-percentile score as covariate and as in-training residualisation", cohorts_used="GSE118553 -> GSE132903",
                 regions_analyzed="Temporal cortex / MTG", model_type="InFold_LogisticCV + composition control",
                 covariates_adjusted="Neuronal marker score, age, sex", sample_size_donors=int(len(lb)),
                 primary_metric="AUC_after_neuronal_residualisation", point_estimate=est, ci_95_lower=lo, ci_95_upper=hi,
                 p_value=out["adj_lr_p"], replicated_in_lockbox=False, status="COMPLETE", notes="Post hoc.")
    return out


if __name__ == "__main__":
    print(run_h3_composition())
