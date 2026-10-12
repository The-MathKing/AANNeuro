"""Checks requested in the third referee report (all post hoc):
  1. n used for the Table 2 age test (donors with a recorded age).
  2. Calibration of a within-sample-rank model (is the H3 calibration failure only about normalization?).
  3. Summary of the repeated-CV, adjusted regional CIs (number excluding zero, median half-width).
  4. E2 multivariable model with the oligodendrocytic score included.
  5. Continuous, amyloid-aware resilience test with the donor neuronal score added (partial Spearman).
  6. Permutation null SD of a 7-vs-53 donor AUC (for comparing interval widths)."""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from src.load import load_gse84422, load_gse118553, load_gse132903
from src.harmonize.cross_platform import compute_sample_ranks, score_gene_signature_singscore
from src.analyses.e2_celltype import compute_cell_type_scores
from src.analyses.h2_resilience import define_resilience_cohort
from src.models.linear_pipeline import InFoldLinearModel
from src.eval.metrics import calc_auc


def _rc_donor_mean(meta, values):
    s = pd.Series(values, index=meta.index)
    z = s.groupby(meta.region).transform(lambda x: (x - x.mean()) / (x.std(ddof=0) or 1.0))
    return z.groupby(meta.donor_id).mean()


def run_r5_checks():
    out = {}
    m118, e118 = load_gse118553(); m132, e132 = load_gse132903(); m84, e84 = load_gse84422()

    # 1. Table 2 age test n
    tc = m118[(m118.region == "Temporal_Cortex") & m118.diagnosis.isin(["CTL", "AD"])]
    dd = tc.groupby("donor_id").agg(dx=("diagnosis", "first"), age=("age", "mean"))
    out["age_test_n_AD"] = int(dd[(dd.dx == "AD")].age.notna().sum())
    out["age_test_n_CTL"] = int(dd[(dd.dx == "CTL")].age.notna().sum())
    out["age_missing_donors"] = int(dd.age.isna().sum())

    # 2. within-sample-rank model calibration
    ytr = (tc.diagnosis == "AD").astype(int).values
    lb = m132[m132.diagnosis.isin(["CTL", "AD"])]
    y = (lb.diagnosis == "AD").astype(int).values
    genes = [g for g in e118.index if g in e132.index]
    Rtr = compute_sample_ranks(e118.loc[genes, tc.index]).T
    Rlb = compute_sample_ranks(e132.loc[genes, lb.index]).T
    mod = InFoldLinearModel("ridge_classifier", top_hvg=1000, seed=42).fit_transform_train(Rtr, ytr)
    p = mod.predict(Rlb)
    out["rank_model"] = {"auc": float(calc_auc(y, p)), "median_p": float(np.median(p)),
                         "sens_0.5": float(((p > .5) & (y == 1)).sum() / (y == 1).sum()),
                         "spec_0.5": float(((p <= .5) & (y == 0)).sum() / (y == 0).sum()),
                         "frac_called_AD": float((p > .5).mean())}

    # 3. regional CI summary (repeated CV, adjusted)
    hr = json.load(open("results/h1_robust.json"))
    ci = hr["variants"]["adjusted"]["regional_ci"]
    excl = [r for r, (lo, hi) in ci.items() if lo > 0 or hi < 0]
    out["regional_adj_n_excl_zero"] = len(excl); out["regional_adj_excl_regions"] = excl
    out["regional_adj_median_halfwidth"] = float(np.median([(hi - lo) / 2 for lo, hi in ci.values()]))
    from src.analyses.h1_regional import PRE_REGISTERED_RANKS
    out["regional_adj_excl_tiers"] = {r: PRE_REGISTERED_RANKS.get(r, 3) for r in excl}

    # 4. E2 with all five scores
    valid = m84.dropna(subset=["braak", "donor_id"])
    cs = compute_cell_type_scores(e84[valid.index])
    dm = valid.groupby("donor_id").agg(braak=("braak", "first"), age=("age", "first"), pmi=("pmi", "first"), sex=("sex", "first"))
    for ct in cs.columns:
        dm[ct] = _rc_donor_mean(valid, cs[ct].values)
    dm["fem"] = (dm.sex == "FEMALE").astype(float)
    cols5 = ["endothelial", "neuronal", "microglial", "astrocytic", "oligodendrocytic", "age", "fem", "pmi"]
    d5 = dm.dropna(subset=cols5)
    f5 = sm.OLS(d5.braak.values, sm.add_constant(d5[cols5].values)).fit()
    out["e2_five_scores"] = {"n": int(len(d5)), "neuronal_beta": float(f5.params[2]), "neuronal_p": float(f5.pvalues[2]),
                             "oligo_beta": float(f5.params[5]), "oligo_p": float(f5.pvalues[5]),
                             "endothelial_beta": float(f5.params[1]), "endothelial_p": float(f5.pvalues[1]),
                             "corr_oligo_neuronal": float(stats.spearmanr(d5.oligodendrocytic, d5.neuronal)[0])}

    # 5. continuous resilience test + neuronal adjustment
    sub = m118[(m118.region == "Temporal_Cortex") & m118.diagnosis.isin(["AsymAD", "AD"])]
    E = e118[sub.index].T; asym = (sub.diagnosis == "AsymAD").values
    t = {}
    for g_ in E.columns:
        tv, _ = stats.ttest_ind(E.loc[asym, g_], E.loc[~asym, g_], equal_var=False)
        if not np.isnan(tv):
            t[g_] = tv
    s = pd.Series(t).sort_values(ascending=False)
    up = [g_ for g_ in s.head(100).index if g_ in e84.index]; dn = [g_ for g_ in s.tail(100).index if g_ in e84.index]
    allv = m84.dropna(subset=["braak", "cdr", "cerad", "age", "sex", "pmi", "donor_id"])
    sc = score_gene_signature_singscore(e84[allv.index], up, dn).values
    cs2 = compute_cell_type_scores(e84[allv.index])
    don = allv.groupby("donor_id").agg(cdr=("cdr", "first"), braak=("braak", "first"), cerad=("cerad", "first"),
                                       age=("age", "first"), sex=("sex", "first"), pmi=("pmi", "first"))
    don["sc"] = _rc_donor_mean(allv, sc); don["neuro"] = _rc_donor_mean(allv, cs2["neuronal"].values)
    Z0 = np.c_[np.ones(len(don)), don.braak, don.cerad, don.age, (don.sex == "FEMALE").astype(float), don.pmi]
    res = lambda v, Z: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    r0 = stats.spearmanr(don.sc, res(don.cdr.values, Z0))
    Z1 = np.c_[Z0, don.neuro]
    r1 = stats.spearmanr(don.sc, res(don.cdr.values, Z1))                      # neuronal added to the CDR residualisation
    r2 = stats.spearmanr(res(don.sc.values, Z1), res(don.cdr.values, Z1))      # full partial: both sides adjusted
    out["continuous"] = {"n": int(len(don)), "rho_base": float(r0[0]), "p_base": float(r0[1]),
                         "rho_neuro_in_cdr_resid": float(r1[0]), "p_neuro_in_cdr_resid": float(r1[1]),
                         "rho_partial_neuro": float(r2[0]), "p_partial_neuro": float(r2[1]),
                         "corr_sc_neuro": float(stats.spearmanr(don.sc, don.neuro)[0])}

    # 6. permutation null SD for 7 vs 53 AUC (Mann-Whitney, no ties)
    n1, n0 = 7, 53
    out["auc_null_sd_7v53"] = float(np.sqrt(n1 * n0 * (n1 + n0 + 1) / 12) / (n1 * n0))
    # 7. calibration of the Table 6 models from their saved lockbox predictions
    ab = json.load(open("results/model_ablation_results.json"))["predictions"]
    yl = np.array(ab["y_lockbox"]); out["table6_calibration"] = {}
    for k in ["baseline_h3", "rank_transform_only"]:
        pp = np.array(ab["preds"][k])
        out["table6_calibration"][k] = {"median_p": float(np.median(pp)), "sens": float(((pp > .5) & (yl == 1)).sum() / (yl == 1).sum()),
                                        "spec": float(((pp <= .5) & (yl == 0)).sum() / (yl == 0).sum())}
    json.dump(out, open("results/r5_checks.json", "w"), indent=2)
    return out


if __name__ == "__main__":
    print(json.dumps(run_r5_checks(), indent=1))
