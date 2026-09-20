"""
Post-hoc resilience overlap test (suggested in Draft 7 review, run in Draft 9).

Question: do genes whose expression tracks cognition (CDR) independent of Braak stage in MSBB (E4) also separate
asymptomatic-AD from symptomatic-AD brains in an independent cohort (GSE118553, temporal cortex)?

Independence: gene sets are defined in MSBB only; GSE118553 is used only to score them.
Direction:    a gene with a NEGATIVE CDR coefficient (higher expression -> better cognition, at fixed Braak) is
              predicted to be HIGHER in AsymAD than in AD; a POSITIVE coefficient predicts LOWER in AsymAD.

Statistic:    S = mean(t_AsymAD_vs_AD over CDR-negative genes) - mean(t_AsymAD_vs_AD over CDR-positive genes).
Null:         AsymAD/AD labels permuted across GSE118553 donors (5,000x); the gene sets stay fixed, so gene-gene
              correlation on the GSE118553 side is preserved. (Gene-label permutation would assume independent genes.)
Gene sets:    E4 primary spec (93 cortical donors) and the all-donor region-centred spec; nominal p<0.01 (and top-N),
              because no gene passes FDR in the primary spec.
Also reports the classic Fisher overlap for comparison with the earlier (near-zero-expected-overlap) test.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from src.load import load_gse84422, load_gse118553
from src.eval.ledger_logger import log_analysis
from src.analyses.e4_dissociation import _partial_p

N_PERM = 5000


def _tstats_matrix(X, labels):
    """Welch-free fast t (pooled variance) for every gene; X (n, G), labels 0/1. Positive = higher in label==1."""
    a, b = X[labels == 1], X[labels == 0]
    na, nb = len(a), len(b)
    sp2 = ((na - 1) * a.var(0, ddof=1) + (nb - 1) * b.var(0, ddof=1)) / (na + nb - 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (a.mean(0) - b.mean(0)) / np.sqrt(sp2 * (1 / na + 1 / nb))


def _e4_gene_table(meta, expr, regions=None, region_center=False):
    valid = meta.dropna(subset=["braak", "cdr", "age", "sex", "pmi", "donor_id"]).copy()
    valid["is_female"] = (valid["sex"] == "FEMALE").astype(float)
    if regions is not None:
        valid = valid[valid["region"].isin(regions)]
    X = expr[valid.index].T.copy()
    if region_center:
        for reg, idx in valid.groupby("region").groups.items():
            sub = X.loc[idx]
            X.loc[idx] = (sub - sub.mean()) / sub.std(ddof=0).replace(0, 1.0)
    X["donor_id"] = valid["donor_id"].values
    Xd = X.groupby("donor_id").mean()
    dm = valid.groupby("donor_id").agg({"braak": "mean", "cdr": "mean", "age": "mean", "is_female": "first", "pmi": "mean"}).loc[Xd.index]
    Xv = np.nan_to_num(Xd.values.astype(float))
    ok = Xv.std(0) >= 1e-5
    cov = dm[["braak", "age", "is_female", "pmi"]].values
    p = np.where(ok, _partial_p(Xv, dm["cdr"].values, cov), 1.0)
    # signed effect: sign of the partial coefficient = sign of partial correlation with CDR
    n = len(dm); Zc = np.column_stack([np.ones(n), cov]); H = Zc @ np.linalg.pinv(Zc)
    xr = Xv - H @ Xv; yr = dm["cdr"].values - H @ dm["cdr"].values
    sign = np.sign(xr.T @ yr)
    return pd.DataFrame({"gene": Xd.columns, "p": p, "sign": sign}).set_index("gene"), int(len(dm))


def _gene_set_test(tab, tstat, gene_index, X118, y118, rng, thr=None, top_n=None, label=""):
    """tab: E4 gene table (p, sign). tstat: observed AsymAD-vs-AD t per gene (Series). Returns dict."""
    common = tab.index.intersection(gene_index)
    t = tab.loc[common]
    if thr is not None:
        sel = t[t["p"] < thr]
    else:
        sel = t.sort_values("p").head(top_n)
    neg = sel[sel["sign"] < 0].index.tolist()   # higher expression -> lower CDR (resilience-consistent if higher in AsymAD)
    pos = sel[sel["sign"] > 0].index.tolist()
    gi = {g: i for i, g in enumerate(gene_index)}
    ineg = [gi[g] for g in neg]; ipos = [gi[g] for g in pos]
    t_obs = tstat.values

    def stat(tv):
        return np.nanmean(tv[ineg]) - np.nanmean(tv[ipos]) if len(ineg) and len(ipos) else np.nan

    S = stat(t_obs)
    null = np.empty(N_PERM)
    for b in range(N_PERM):
        null[b] = stat(_tstats_matrix(X118, rng.permutation(y118)))
    p_perm = float((np.sum(null >= S) + 1) / (N_PERM + 1))
    # directional means separately
    m_neg, m_pos = float(np.nanmean(t_obs[ineg])), float(np.nanmean(t_obs[ipos]))
    # sign concordance among genes nominally significant (p<0.05) in GSE118553 too
    sig118 = np.abs(t_obs) > 2.0
    concord = []
    for g_list, expected in [(ineg, 1), (ipos, -1)]:
        for i in g_list:
            if sig118[i]:
                concord.append(np.sign(t_obs[i]) == expected)
    return {
        "label": label, "n_genes_neg_cdr": len(neg), "n_genes_pos_cdr": len(pos),
        "mean_t_asymAD_vs_AD_in_negCDR_set": m_neg, "mean_t_asymAD_vs_AD_in_posCDR_set": m_pos,
        "statistic_S": float(S), "null_mean": float(np.nanmean(null)), "null_sd": float(np.nanstd(null)),
        "perm_p_one_sided": p_perm,
        "n_overlap_significant_in_118553": int(len(concord)),
        "frac_sign_concordant_among_significant": float(np.mean(concord)) if concord else float("nan"),
    }


def run_resilience_overlap():
    print("=== POST-HOC RESILIENCE OVERLAP: E4 COGNITION GENES vs GSE118553 AsymAD-vs-AD ===")
    os.makedirs("results", exist_ok=True)
    m84, e84 = load_gse84422()
    m118, e118 = load_gse118553()

    sub = m118[(m118["region"] == "Temporal_Cortex") & (m118["diagnosis"].isin(["AsymAD", "AD"]))]
    X118_df = e118[sub.index].T
    X118 = X118_df.values.astype(float)
    y118 = (sub["diagnosis"] == "AsymAD").astype(int).values     # 1 = AsymAD
    gene_index = X118_df.columns
    t_obs = pd.Series(_tstats_matrix(X118, y118), index=gene_index)
    print(f"GSE118553 TC: {int(y118.sum())} AsymAD vs {int((1 - y118).sum())} AD; {len(gene_index)} genes")

    rng = np.random.default_rng(42)
    specs = {
        "primary_93_cortical_donors": dict(regions=["BM9_DLPFC", "BM10_FP", "BM22_STG", "BM36_PHG", "BM20_ITG"], region_center=False),
        "all_117_donors_region_centred": dict(regions=None, region_center=True),
    }
    out = {"n_asymAD": int(y118.sum()), "n_AD": int((1 - y118).sum()), "n_perm": N_PERM, "specs": {}}
    for name, kw in specs.items():
        tab, n_d = _e4_gene_table(m84, e84, **kw)
        out["specs"][name] = {"n_donors": n_d, "sets": {}}
        for lab, arg in [("p<0.01", dict(thr=0.01)), ("p<0.05", dict(thr=0.05)), ("top300", dict(top_n=300))]:
            r = _gene_set_test(tab, t_obs, gene_index, X118, y118, rng, label=f"{name}:{lab}", **arg)
            out["specs"][name]["sets"][lab] = r
            print(f"  {name:32s} {lab:7s} negCDR n={r['n_genes_neg_cdr']:4d} meant={r['mean_t_asymAD_vs_AD_in_negCDR_set']:+.3f} | posCDR n={r['n_genes_pos_cdr']:4d} meant={r['mean_t_asymAD_vs_AD_in_posCDR_set']:+.3f} | S={r['statistic_S']:+.3f} perm p={r['perm_p_one_sided']:.4f}")

    # classic Fisher overlap for comparison: top-100 AsymAD-up (as in the original H2 code) vs E4 CDR-negative p<0.01
    tab_p, _ = _e4_gene_table(m84, e84, **specs["primary_93_cortical_donors"])
    common = tab_p.index.intersection(gene_index)
    neg = set(tab_p.loc[common][(tab_p.loc[common]["p"] < 0.01) & (tab_p.loc[common]["sign"] < 0)].index)
    up_asym = set(t_obs.loc[common].sort_values(ascending=False).head(100).index)
    ov = len(neg & up_asym)
    tbl = [[ov, len(up_asym) - ov], [len(neg) - ov, len(common) - len(up_asym) - len(neg) + ov]]
    orr, pf = stats.fisher_exact(tbl, alternative="greater")
    out["fisher_top100_asymAD_up_vs_negCDR_p01"] = {"overlap": ov, "expected_by_chance": len(neg) * len(up_asym) / len(common), "n_negCDR": len(neg), "odds_ratio": float(orr), "p": float(pf)}
    print(f"  Fisher (top-100 AsymAD-up vs CDR-negative p<0.01): overlap {ov}, expected {out['fisher_top100_asymAD_up_vs_negCDR_p01']['expected_by_chance']:.1f}, OR {orr:.2f}, p={pf:.3g}")

    with open("results/resilience_overlap.json", "w") as f:
        json.dump(out, f, indent=2)

    main = out["specs"]["primary_93_cortical_donors"]["sets"]["p<0.01"]
    log_analysis(
        analysis_id="RESIL_OVL_001", category="EXPLORATORY_POSTHOC",
        hypothesis_or_idea="E4 cognition-tracking genes vs GSE118553 AsymAD-vs-AD (post hoc, Draft 7 suggestion)",
        description="Directional concordance of MSBB CDR|Braak genes with AsymAD-vs-AD t-statistics in GSE118553 TC; AsymAD/AD label permutation null",
        cohorts_used="GSE84422_MSBB -> GSE118553", regions_analyzed="MSBB 5 cortical regions / GSE118553 TC",
        model_type="Gene-set concordance, label-permutation null", covariates_adjusted="Braak, Age, Sex, PMI (MSBB side only)",
        sample_size_donors=int(y118.size), primary_metric="S_meanT_negCDR_minus_posCDR", point_estimate=main["statistic_S"],
        ci_95_lower=float("nan"), ci_95_upper=float("nan"), p_value=main["perm_p_one_sided"], replicated_in_lockbox=False,
        status="COMPLETE", notes="Gene sets from E4 nominal p<0.01 (primary spec). Post hoc; hypothesis-generating.")
    return out


if __name__ == "__main__":
    run_resilience_overlap()
