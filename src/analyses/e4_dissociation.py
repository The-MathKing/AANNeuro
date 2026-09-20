"""
Exploratory Analysis E4: Pathology vs Cognition Transcriptomic Dissociation.
In Mount Sinai Brain Bank (GSE84422):
1. Identifies genes associated with Clinical Dementia Rating (CDR) after controlling for Braak stage, age, sex, and PMI.
   -> Cognition-specific / Resilience gene candidates.
2. Identifies genes associated with Braak stage after controlling for CDR, age, sex, and PMI.
   -> Pure neuropathology / Tangle gene candidates.
"""

import os
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from src.load import load_gse84422
from src.eval.ledger_logger import log_analysis


from scipy import stats as _st

N_PERM_NULL = 1000


def _partial_p(X, y, Z):
    """Two-sided p-value of the coefficient on each column of X in OLS  y ~ [const, x_g, Z]  (Frisch-Waugh-Lovell).
    X: (n, G) genes; y: (n,) or (n, B) outcomes; Z: (n, q) covariates WITHOUT constant. Returns (G,) or (G, B)."""
    n = X.shape[0]
    Zc = np.column_stack([np.ones(n), Z])
    H = Zc @ np.linalg.pinv(Zc)
    Xr = X - H @ X
    y2 = y if y.ndim == 2 else y[:, None]
    yr = y2 - H @ y2
    df = n - Zc.shape[1] - 1
    num = Xr.T @ yr
    den = np.sqrt((Xr ** 2).sum(0))[:, None] * np.sqrt((yr ** 2).sum(0))[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(den > 0, num / den, 0.0)
        t = r * np.sqrt(df / np.clip(1 - r ** 2, 1e-12, None))
    p = 2 * _st.t.sf(np.abs(t), df)
    return p if y.ndim == 2 else p[:, 0]


def _permute_within_strata(y, strata, rng, n_perm):
    """B permutations of y in which values are shuffled only among donors that share the same stratum label."""
    out = np.empty((len(y), n_perm))
    groups = [np.where(strata == v)[0] for v in np.unique(strata)]
    for b in range(n_perm):
        yb = y.copy()
        for g in groups:
            if len(g) > 1:
                yb[g] = y[rng.permutation(g)]
        out[:, b] = yb
    return out


def _null_counts(X, y, cov, strata, rng, n_perm=N_PERM_NULL, thr=0.01):
    """Null distribution of the number of genes with p < thr when y carries no information beyond its stratum."""
    Yp = _permute_within_strata(y, strata, rng, n_perm)
    P = _partial_p(X, Yp, cov)
    return (P < thr).sum(axis=0)


def run_e4_analysis():
    print("=== RUNNING EXPLORATORY ANALYSIS E4: PATHOLOGY VS COGNITION DISSOCIATION ===")
    os.makedirs("results", exist_ok=True)

    meta_msbb, expr_msbb = load_gse84422()
    # Filter to cortical samples with complete Braak, CDR, Age, Sex, PMI
    valid = meta_msbb.dropna(subset=["braak", "cdr", "age", "sex", "pmi", "donor_id"]).copy()
    valid["braak"] = valid["braak"].astype(float)
    valid["cdr"] = valid["cdr"].astype(float)
    valid["is_female"] = (valid["sex"] == "FEMALE").astype(float)
    valid["pmi"] = valid["pmi"].astype(float)
    valid["age"] = valid["age"].astype(float)

    # Focus on Frontal / Temporal cortex regions with highest sample depth
    cortical = valid[valid["region"].isin(["BM9_DLPFC", "BM10_FP", "BM22_STG", "BM36_PHG", "BM20_ITG"])].copy()
    
    # Aggregate to Donor-Level (one mean profile per donor to strictly prevent pseudoreplication)
    donor_meta = cortical.groupby("donor_id").agg({
        "braak": "mean",
        "cdr": "mean",
        "age": "mean",
        "is_female": "first",
        "pmi": "mean"
    })
    
    donor_expr_dict = {}
    for d_id in donor_meta.index:
        s_ids = cortical[cortical["donor_id"] == d_id].index
        donor_expr_dict[d_id] = expr_msbb[s_ids].mean(axis=1)
        
    sub_expr = pd.DataFrame(donor_expr_dict).T  # donors x genes
    
    print(f"Donor-Level Cortical Cohort: {len(donor_meta)} unique donors across {sub_expr.shape[1]} genes.")

    genes = sub_expr.columns.tolist()
    
    # Pre-build design matrices
    cov_cdr_model = donor_meta[["braak", "age", "is_female", "pmi"]].values
    y_cdr = donor_meta["cdr"].values
    
    cov_braak_model = donor_meta[["cdr", "age", "is_female", "pmi"]].values
    y_braak = donor_meta["braak"].values

    cdr_specific_pvals = []
    cdr_specific_betas = []
    braak_specific_pvals = []
    braak_specific_betas = []

    for g in genes:
        x_gene = sub_expr[g].values
        if np.std(x_gene) < 1e-5:
            cdr_specific_pvals.append(1.0)
            cdr_specific_betas.append(0.0)
            braak_specific_pvals.append(1.0)
            braak_specific_betas.append(0.0)
            continue

        # Model 1: CDR ~ Gene + Braak + Age + Sex + PMI
        X1 = sm.add_constant(np.column_stack([x_gene, cov_cdr_model]), has_constant='add')
        try:
            res1 = sm.OLS(y_cdr, X1).fit()
            cdr_specific_betas.append(float(res1.params[1]))
            cdr_specific_pvals.append(float(res1.pvalues[1]))
        except Exception:
            cdr_specific_betas.append(0.0)
            cdr_specific_pvals.append(1.0)

        # Model 2: Braak ~ Gene + CDR + Age + Sex + PMI
        X2 = sm.add_constant(np.column_stack([x_gene, cov_braak_model]), has_constant='add')
        try:
            res2 = sm.OLS(y_braak, X2).fit()
            braak_specific_betas.append(float(res2.params[1]))
            braak_specific_pvals.append(float(res2.pvalues[1]))
        except Exception:
            braak_specific_betas.append(0.0)
            braak_specific_pvals.append(1.0)

    # ---- Verification: vectorised regression must reproduce the loop above ----
    Xg = sub_expr.values.astype(float)
    ok = Xg.std(axis=0) >= 1e-5
    p_cdr_vec = np.where(ok, _partial_p(Xg, y_cdr, cov_cdr_model), 1.0)
    p_braak_vec = np.where(ok, _partial_p(Xg, y_braak, cov_braak_model), 1.0)
    assert np.allclose(p_cdr_vec, np.array(cdr_specific_pvals), atol=1e-8), "vectorised CDR p-values do not match OLS loop"
    assert np.allclose(p_braak_vec, np.array(braak_specific_pvals), atol=1e-8), "vectorised Braak p-values do not match OLS loop"

    # ---- Null baseline: permute CDR within Braak strata, Braak within CDR strata ----
    rng_null = np.random.default_rng(42)
    null_cdr = _null_counts(Xg, y_cdr, cov_cdr_model, donor_meta["braak"].values, rng_null)
    null_braak = _null_counts(Xg, y_braak, cov_braak_model, donor_meta["cdr"].values, rng_null)
    obs_cdr = int((p_cdr_vec < 0.01).sum()); obs_braak = int((p_braak_vec < 0.01).sum())
    null_ratio = null_cdr / np.maximum(null_braak, 1)
    n_test_genes = int(ok.sum())
    braak_cdr_rho = float(_st.spearmanr(donor_meta["braak"], donor_meta["cdr"])[0])
    print(f"Null (within-stratum permutation, {N_PERM_NULL} perms): CDR|Braak count mean={null_cdr.mean():.0f} sd={null_cdr.std():.0f} max={null_cdr.max()}; "
          f"Braak|CDR mean={null_braak.mean():.0f} sd={null_braak.std():.0f} max={null_braak.max()}; observed {obs_cdr} vs {obs_braak}")
    print(f"Naive expectation at p<0.01 across {n_test_genes} genes: {0.01 * n_test_genes:.0f}; null ratio median={np.median(null_ratio):.2f} (95th pct {np.percentile(null_ratio, 95):.2f}); observed ratio {obs_cdr / max(obs_braak, 1):.2f}")

    # ---- Sensitivity: all donors with valid phenotype, region-centred expression averaged over every region sampled ----
    all_samples = valid.copy()
    Xall = expr_msbb[all_samples.index].T.copy()
    for reg, idx in all_samples.groupby("region").groups.items():
        sub = Xall.loc[idx]
        Xall.loc[idx] = (sub - sub.mean()) / sub.std(ddof=0).replace(0, 1.0)
    Xall["donor_id"] = all_samples["donor_id"].values
    Xall_d = Xall.groupby("donor_id").mean()
    dm_all = all_samples.groupby("donor_id").agg({"braak": "mean", "cdr": "mean", "age": "mean", "is_female": "first", "pmi": "mean"}).loc[Xall_d.index]
    Xa = Xall_d.values.astype(float); oka = np.nan_to_num(Xa.std(axis=0)) >= 1e-5
    Xa = np.nan_to_num(Xa)
    pa_cdr = np.where(oka, _partial_p(Xa, dm_all["cdr"].values, dm_all[["braak", "age", "is_female", "pmi"]].values), 1.0)
    pa_braak = np.where(oka, _partial_p(Xa, dm_all["braak"].values, dm_all[["cdr", "age", "is_female", "pmi"]].values), 1.0)
    cov_c_all = dm_all[["braak", "age", "is_female", "pmi"]].values; cov_b_all = dm_all[["cdr", "age", "is_female", "pmi"]].values
    rng_s = np.random.default_rng(43)
    nc_all = _null_counts(Xa, dm_all["cdr"].values, cov_c_all, dm_all["braak"].values, rng_s)
    nb_all = _null_counts(Xa, dm_all["braak"].values, cov_b_all, dm_all["cdr"].values, rng_s)
    obs_c_all = int((pa_cdr < 0.01).sum()); obs_b_all = int((pa_braak < 0.01).sum())
    top_all = pd.DataFrame({"gene": Xall_d.columns, "p": pa_cdr}).sort_values("p").head(25)
    top_all["beta_sign"] = [float(np.sign(np.corrcoef(Xa[:, list(Xall_d.columns).index(g)], dm_all["cdr"].values)[0, 1])) for g in top_all["gene"]]
    sens_all = {"null_cdr_mean": float(nc_all.mean()), "null_cdr_sd": float(nc_all.std()), "null_cdr_p95": float(np.percentile(nc_all, 95)),
                "null_braak_mean": float(nb_all.mean()), "null_braak_sd": float(nb_all.std()), "null_braak_p95": float(np.percentile(nb_all, 95)),
                "perm_p_cdr": float((np.sum(nc_all >= obs_c_all) + 1) / (N_PERM_NULL + 1)),
                "perm_p_braak": float((np.sum(nb_all >= obs_b_all) + 1) / (N_PERM_NULL + 1)),
                "perm_p_ratio": float((np.sum(nc_all / np.maximum(nb_all, 1) >= obs_c_all / max(obs_b_all, 1)) + 1) / (N_PERM_NULL + 1)),
                "top_cdr_genes": top_all.to_dict(orient="records"),
                "n_donors": int(len(dm_all)), "n_cdr_p01": int((pa_cdr < 0.01).sum()), "n_braak_p01": int((pa_braak < 0.01).sum()),
                "n_cdr_fdr05": int((multipletests(pa_cdr, method="fdr_bh")[1] < 0.05).sum()),
                "n_braak_fdr05": int((multipletests(pa_braak, method="fdr_bh")[1] < 0.05).sum())}
    print(f"Sensitivity, all {sens_all['n_donors']} donors (region-centred, every region): CDR|Braak {sens_all['n_cdr_p01']} vs Braak|CDR {sens_all['n_braak_p01']} at p<0.01")

    # BH-FDR corrections
    _, q_cdr, _, _ = multipletests(cdr_specific_pvals, alpha=0.05, method='fdr_bh')
    _, q_braak, _, _ = multipletests(braak_specific_pvals, alpha=0.05, method='fdr_bh')

    df_dissoc = pd.DataFrame({
        "gene": genes,
        "cdr_beta": cdr_specific_betas,
        "cdr_pvalue": cdr_specific_pvals,
        "cdr_qvalue": q_cdr,
        "braak_beta": braak_specific_betas,
        "braak_pvalue": braak_specific_pvals,
        "braak_qvalue": q_braak
    })

    top_cdr_p01 = df_dissoc[df_dissoc["cdr_pvalue"] < 0.01].sort_values("cdr_pvalue")
    top_braak_p01 = df_dissoc[df_dissoc["braak_pvalue"] < 0.01].sort_values("braak_pvalue")
    top_cdr_fdr05 = df_dissoc[df_dissoc["cdr_qvalue"] < 0.05].sort_values("cdr_pvalue")
    top_braak_fdr05 = df_dissoc[df_dissoc["braak_qvalue"] < 0.05].sort_values("braak_pvalue")

    print(f"\nDonor-Level Dissociation Results (N = {len(donor_meta)} donors):")
    print(f"  Cognition-Specific Genes (CDR controlling for Braak, nominal p < 0.01): {len(top_cdr_p01)} (FDR<0.05: {len(top_cdr_fdr05)})")
    print(f"  Pathology-Specific Genes (Braak controlling for CDR, nominal p < 0.01): {len(top_braak_p01)} (FDR<0.05: {len(top_braak_fdr05)})")

    results = {
        "analysis": "E4_Pathology_vs_Cognition_Dissociation",
        "n_samples": int(len(cortical)),
        "n_donors": int(len(donor_meta)),
        "n_cdr_specific_p01": int(len(top_cdr_p01)),
        "n_braak_specific_p01": int(len(top_braak_p01)),
        "n_donors_with_complete_phenotype": int(valid["donor_id"].nunique()),
        "n_donors_excluded_no_cortical_sample": int(valid["donor_id"].nunique() - len(donor_meta)),
        "cortical_regions_used": ["BM9_DLPFC", "BM10_FP", "BM22_STG", "BM36_PHG", "BM20_ITG"],
        "n_genes_tested": n_test_genes,
        "braak_cdr_spearman_rho_among_donors": braak_cdr_rho,
        "naive_expected_p01": float(0.01 * n_test_genes),
        "null_cdr_mean": float(null_cdr.mean()), "null_cdr_sd": float(null_cdr.std()),
        "null_cdr_p95": float(np.percentile(null_cdr, 95)), "null_cdr_max": int(null_cdr.max()),
        "null_braak_mean": float(null_braak.mean()), "null_braak_sd": float(null_braak.std()),
        "null_braak_p95": float(np.percentile(null_braak, 95)), "null_braak_max": int(null_braak.max()),
        "perm_p_cdr": float((np.sum(null_cdr >= obs_cdr) + 1) / (N_PERM_NULL + 1)),
        "perm_p_braak": float((np.sum(null_braak >= obs_braak) + 1) / (N_PERM_NULL + 1)),
        "observed_ratio": float(obs_cdr / max(obs_braak, 1)),
        "null_ratio_median": float(np.median(null_ratio)), "null_ratio_p95": float(np.percentile(null_ratio, 95)),
        "null_ratio_max": float(null_ratio.max()),
        "perm_p_ratio": float((np.sum(null_ratio >= obs_cdr / max(obs_braak, 1)) + 1) / (N_PERM_NULL + 1)),
        "n_permutations": N_PERM_NULL,
        "sensitivity_all_donors": sens_all,
        "n_cdr_specific_fdr05": int(len(top_cdr_fdr05)),
        "n_braak_specific_fdr05": int(len(top_braak_fdr05)),
        "top_cdr_specific_genes": top_cdr_p01.head(20).to_dict(orient="records"),
        "top_braak_specific_genes": top_braak_p01.head(20).to_dict(orient="records")
    }

    with open("results/e4_dissociation.json", "w") as f:
        json.dump(results, f, indent=2)

    log_analysis(
        analysis_id="E4_DISSOC_001",
        category="EXPLORATORY",
        hypothesis_or_idea="E4 Pathology vs Cognition Dissociation",
        description="Donor-level multivariable regression isolating CDR-specific vs Braak-specific transcriptomic drivers",
        cohorts_used="GSE84422_MSBB",
        regions_analyzed="Cortical association regions (Donor Mean)",
        model_type="OLS_Multivariable_DonorLevel",
        covariates_adjusted="Braak Stage / CDR, Age, Sex, PMI",
        sample_size_donors=len(donor_meta),
        primary_metric="N_CDR_Specific_Genes_P01",
        point_estimate=len(top_cdr_p01),
        ci_95_lower=float('nan'),
        ci_95_upper=float('nan'),
        p_value=float('nan'),
        replicated_in_lockbox=False,
        status="COMPLETE",
        notes=f"Identified {len(top_cdr_p01)} cognition-specific vs {len(top_braak_p01)} pathology-specific genes at nominal p < 0.01 (donor-level N={len(donor_meta)})."
    )

    return results


if __name__ == "__main__":
    run_e4_analysis()
