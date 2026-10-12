"""Donor-level, covariate-adjusted version of the post hoc resilience-overlap test.
The earlier version scored 32 AsymAD vs 52 AD temporal-cortex SAMPLES; several donors contribute more than one sample.
Here GSE118553 temporal-cortex samples are averaged to one profile per donor, and
 (i) unadjusted donor-level Welch t per gene, permutation of donor labels;
 (ii) adjusted: per-gene partial t for AsymAD-vs-AD controlling for age and neuronal marker score (Frisch-Waugh),
      permutation of donor labels (residualised expression is computed once; the label residual is recomputed each draw).
Gene sets are the MSBB E4 gene sets (defined without GSE118553): nominal p<0.01 primary spec (93 cortical donors).
Statistic S = mean t over CDR-negative genes minus mean t over CDR-positive genes."""
import json
import numpy as np
import pandas as pd
from scipy import stats
from src.load import load_gse84422, load_gse118553
from src.analyses.resilience_overlap import _e4_gene_table
from src.harmonize.cross_platform import compute_sample_ranks
from src.eval.ledger_logger import log_analysis

N_PERM = 5000
NEURONAL = ["RBFOX3", "SNAP25", "SYP", "NEFL", "SLC17A7", "GAD1", "MAP2", "CAMK2A"]


def _t_adj(Xr, y, Zc):
    """partial t of y on each column of Xr (already residualised on Z), df = n - ncov - 2"""
    H = Zc @ np.linalg.pinv(Zc)
    yr = y - H @ y
    df = len(y) - Zc.shape[1] - 1
    num = Xr.T @ yr
    den = np.sqrt((Xr ** 2).sum(0)) * np.sqrt((yr ** 2).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(den > 0, num / den, 0.0)
        return r * np.sqrt(df / np.clip(1 - r ** 2, 1e-12, None))


def run_resilience_overlap_donor():
    m84, e84 = load_gse84422()
    m118, e118 = load_gse118553()
    sub = m118[(m118.region == "Temporal_Cortex") & m118.diagnosis.isin(["AsymAD", "AD"])].copy()
    ncount = sub.groupby("donor_id").size()
    mk = [g for g in NEURONAL if g in e118.index]
    sub["neuro"] = compute_sample_ranks(e118[sub.index]).loc[mk].mean(0).values
    E = e118[sub.index].T
    E["donor_id"] = sub.donor_id.values
    Xd = E.groupby("donor_id").mean()
    dm = sub.groupby("donor_id").agg(dx=("diagnosis", "first"), age=("age", "mean"), neuro=("neuro", "mean"))
    dm["age"] = dm.age.fillna(dm.age.median())
    assert (sub.groupby("donor_id").diagnosis.nunique() == 1).all()
    y = (dm.dx == "AsymAD").astype(int).values
    X = Xd.loc[dm.index].values.astype(float)
    genes = np.array(Xd.columns)
    ok = X.std(0) > 1e-8
    # (i) unadjusted donor-level t
    def tstat(Xm, yy):
        a, b = Xm[yy == 1], Xm[yy == 0]
        na, nb = len(a), len(b)
        sp = ((na - 1) * a.var(0, ddof=1) + (nb - 1) * b.var(0, ddof=1)) / (na + nb - 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            return (a.mean(0) - b.mean(0)) / np.sqrt(sp * (1 / na + 1 / nb))
    t_un = tstat(X, y)
    # (ii) adjusted
    Z = np.c_[np.ones(len(y)), dm.age.values, dm.neuro.values]
    Hz = Z @ np.linalg.pinv(Z)
    Xr = X - Hz @ X
    t_ad = _t_adj(Xr, y.astype(float), Z)

    rng = np.random.default_rng(42)
    specs = {"primary_93_cortical_donors": dict(regions=["BM9_DLPFC", "BM10_FP", "BM22_STG", "BM36_PHG", "BM20_ITG"], region_center=False),
             "all_117_donors_region_centred": dict(regions=None, region_center=True)}
    out = {"n_asymAD_donors": int(y.sum()), "n_AD_donors": int((1 - y).sum()),
           "n_asymAD_samples": int((sub.diagnosis == "AsymAD").sum()), "n_AD_samples": int((sub.diagnosis == "AD").sum()),
           "n_donors_with_multiple_samples": int((ncount > 1).sum()), "neuronal_markers_used": mk, "n_perm": N_PERM, "specs": {}}
    gi = {g: i for i, g in enumerate(genes)}
    for name, kw in specs.items():
        tab, n_d = _e4_gene_table(m84, e84, **kw)
        t = tab.loc[tab.index.intersection(genes)]
        sel = t[t.p < 0.01]
        ineg = [gi[g] for g in sel[sel.sign < 0].index if ok[gi[g]]]
        ipos = [gi[g] for g in sel[sel.sign > 0].index if ok[gi[g]]]
        res = {"n_neg": len(ineg), "n_pos": len(ipos)}
        for label, tv, fn in [("unadjusted", t_un, lambda yy: tstat(X, yy)),
                              ("adjusted", t_ad, lambda yy: _t_adj(Xr, yy.astype(float), Z))]:
            S = np.nanmean(tv[ineg]) - np.nanmean(tv[ipos])
            null = np.empty(N_PERM)
            for b in range(N_PERM):
                tp = fn(rng.permutation(y))
                null[b] = np.nanmean(tp[ineg]) - np.nanmean(tp[ipos])
            res[label] = {"S": float(S), "mean_t_neg": float(np.nanmean(tv[ineg])), "mean_t_pos": float(np.nanmean(tv[ipos])),
                          "null_sd": float(np.nanstd(null)), "perm_p": float((np.sum(null >= S) + 1) / (N_PERM + 1))}
        out["specs"][name] = res
        print(name, res, flush=True)
    json.dump(out, open("results/resilience_overlap_donor.json", "w"), indent=2)
    r = out["specs"]["primary_93_cortical_donors"]["adjusted"]
    log_analysis(analysis_id="RESIL_OVL_002", category="EXPLORATORY_POSTHOC",
                 hypothesis_or_idea="Resilience overlap at donor level, adjusted for age and neuronal score (supersedes RESIL_OVL_001)",
                 description="GSE118553 TC averaged per donor; donor-label permutation; partial t controlling age and neuronal marker score",
                 cohorts_used="GSE84422_MSBB -> GSE118553", regions_analyzed="MSBB 5 cortical regions / GSE118553 TC",
                 model_type="Gene-set concordance, donor permutation", covariates_adjusted="Age, neuronal marker score (GSE118553 side)",
                 sample_size_donors=int(len(y)), primary_metric="S_adjusted", point_estimate=r["S"], ci_95_lower=float("nan"),
                 ci_95_upper=float("nan"), p_value=r["perm_p"], replicated_in_lockbox=False, status="COMPLETE", notes="Post hoc.")
    return out


if __name__ == "__main__":
    run_resilience_overlap_donor()
