"""H3 diagnostics requested in review:
  (a) what the frozen model did with features absent from the lockbox (109 of 1,000) and whether it matters;
  (b) donor-level demographics of the H3 training set (the sample-level table double-counted donors who contribute
      several temporal-cortex samples).
The frozen model (InFoldLinearModel.predict) reindexes missing columns with raw value 0.0, which after z-scoring is a
large constant negative offset for those features. Because it is a constant across lockbox donors it cannot change
AUC; it does shift probabilities. Variants: as-run, training-mean imputation (z = 0), and refit on shared genes only."""
import json
import numpy as np
import pandas as pd
from scipy import stats
from src.load import load_gse118553, load_gse132903
from src.models.linear_pipeline import InFoldLinearModel
from src.eval.metrics import calc_auc


def run_h3_missing_features():
    m1, e1 = load_gse118553()
    m2, e2 = load_gse132903()
    sub = m1[(m1.region == "Temporal_Cortex") & m1.diagnosis.isin(["CTL", "AD"])]
    ytr = (sub.diagnosis == "AD").astype(int).values
    mod = InFoldLinearModel("ridge_classifier", top_hvg=1000, seed=42).fit_transform_train(e1[sub.index].T, ytr)
    lb = m2[m2.diagnosis.isin(["CTL", "AD"])]
    y = (lb.diagnosis == "AD").astype(int).values
    X = e2[lb.index].T
    miss = [g for g in mod.selected_genes if g not in e2.index]
    as_run = mod.predict(X)
    mu = pd.Series(mod.scaler.mean_, index=mod.selected_genes)
    Xi = X.reindex(columns=mod.selected_genes)
    for g in miss:
        Xi[g] = mu[g]
    imputed = mod.model.predict_proba(mod.scaler.transform(Xi.values))[:, 1]
    pres = [g for g in mod.selected_genes if g in e2.index]
    m3 = InFoldLinearModel("ridge_classifier", top_hvg=None, seed=42).fit_transform_train(e1[sub.index].T[pres], ytr)
    refit = m3.predict(X[pres])
    zfill = float(np.mean([(0 - mod.scaler.mean_[i]) / mod.scaler.scale_[i] for i, g in enumerate(mod.selected_genes) if g in miss]))

    # donor-level demographics (training)
    dd = sub.groupby("donor_id").agg(dx=("diagnosis", "first"), age=("age", "mean"), sex=("sex", "first"))
    ad, ct = dd[dd.dx == "AD"], dd[dd.dx == "CTL"]
    chi = stats.chi2_contingency(pd.crosstab(dd.dx, dd.sex))
    from src.load.geo_parser import parse_series_matrix
    _, ex_tr = parse_series_matrix("data/raw/GSE118553_series_matrix.txt.gz", load_expression=True)
    _, ex_lb = parse_series_matrix("data/raw/GSE132903_series_matrix.txt.gz", load_expression=True)
    probe_audit = {"probes_training_deposit": int(ex_tr.shape[0]), "probes_lockbox_deposit": int(ex_lb.shape[0]),
                   "probes_only_in_training": int(len(set(ex_tr.index) - set(ex_lb.index))),
                   "probes_only_in_lockbox": int(len(set(ex_lb.index) - set(ex_tr.index)))}
    out = {
        "probe_audit": probe_audit,
        "n_features": int(len(mod.selected_genes)), "n_missing": int(len(miss)), "missing_examples": miss[:12],
        "n_y_linked_missing": int(sum(g in {"KDM5D", "TMSB4Y", "UTY", "DDX3Y", "EIF1AY", "ZFY", "RPS4Y1"} for g in miss)),
        "auc_as_run": float(calc_auc(y, as_run)), "auc_mean_imputed": float(calc_auc(y, imputed)), "auc_refit_shared_genes": float(calc_auc(y, refit)),
        "mean_z_given_to_missing": zfill,
        "median_p_as_run": float(np.median(as_run)), "median_p_imputed": float(np.median(imputed)),
        "sens_imputed_0.5": float(((imputed > .5) & (y == 1)).sum() / (y == 1).sum()),
        "spec_imputed_0.5": float(((imputed <= .5) & (y == 0)).sum() / (y == 0).sum()),
        "train_expr_min": float(e1.values.min()), "train_expr_max": float(e1.values.max()),
        "lockbox_expr_min": float(e2.values.min()), "lockbox_expr_max": float(e2.values.max()),
        "train_expr_median": float(np.median(e1.values)), "lockbox_expr_median": float(np.median(e2.values)),
        "demographics_training_donor_level": {
            "n_samples": int(len(sub)), "n_donors": int(len(dd)),
            "n_AD": int(len(ad)), "n_CTL": int(len(ct)),
            "age_AD_mean": float(ad.age.mean()), "age_AD_sd": float(ad.age.std()),
            "age_CTL_mean": float(ct.age.mean()), "age_CTL_sd": float(ct.age.std()),
            "age_diff_welch_p": float(stats.ttest_ind(ad.age.dropna(), ct.age.dropna(), equal_var=False)[1]),
            "female_AD": int((ad.sex == "FEMALE").sum()), "female_CTL": int((ct.sex == "FEMALE").sum()),
            "sex_chi2_p": float(chi[1]),
        },
    }
    json.dump(out, open("results/h3_missing_features.json", "w"), indent=2)
    return out


if __name__ == "__main__":
    print(run_h3_missing_features())
