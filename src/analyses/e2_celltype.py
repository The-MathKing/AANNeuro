"""
Exploratory Analysis E2: Cell-Type Composition Deconvolution & Neurovascular Integrity.
1. Computes cell-type surrogate scores using small hand-curated canonical marker panels (not the BRETIGEA resource):
   - Neurons: RBFOX3, SNAP25, SYP, NEFL, SLC17A7, GAD1, MAP2, CAMK2A
   - Astrocytes: GFAP, ALDH1L1, AQP4, SLC1A2, SLC1A3
   - Microglia: AIF1, CD68, CX3CR1, ITGAM, P2RY12, TMEM119
   - Oligodendrocytes: MBP, MOG, OLIG2, SOX10, PLP1
   - Endothelial / Neurovascular: CLDN5, PECAM1, VWF, CD34, CDH5, FLT1
2. Tests whether disease progression signal survives adjustment for neuronal loss.
3. Tests endothelial/blood-brain barrier integrity scores directly against Braak stages in MSBB.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from src.load import load_gse84422, load_gse118553
from src.eval.metrics import calc_spearman, calc_pearson
from src.eval.bootstrap import donor_bootstrap_ci
from src.eval.ledger_logger import log_analysis
from src.harmonize.cross_platform import compute_sample_ranks

# Validated brain cell-type marker sets
MARKER_SETS = {
    "neuronal": ["RBFOX3", "SNAP25", "SYP", "NEFL", "SLC17A7", "GAD1", "MAP2", "CAMK2A"],
    "astrocytic": ["GFAP", "ALDH1L1", "AQP4", "SLC1A2", "SLC1A3"],
    "microglial": ["AIF1", "CD68", "CX3CR1", "ITGAM", "P2RY12", "TMEM119"],
    "oligodendrocytic": ["MBP", "MOG", "OLIG2", "SOX10", "PLP1"],
    "endothelial": ["CLDN5", "PECAM1", "VWF", "CD34", "CDH5", "FLT1"]
}


def compute_cell_type_scores(expr_df):
    """
    Compute mean rank percentiles of marker genes per cell type.
    expr_df: (genes x samples)
    Returns: DataFrame of samples x cell types
    """
    ranks = compute_sample_ranks(expr_df)
    scores = {}
    
    for cell_type, genes in MARKER_SETS.items():
        present = [g for g in genes if g in ranks.index]
        if present:
            scores[cell_type] = ranks.loc[present].mean(axis=0)
        else:
            scores[cell_type] = pd.Series(0.5, index=expr_df.columns)
            
    return pd.DataFrame(scores)


def _donor_level_scores(meta, scores, region_center=True):
    """One row per donor. Region-centering (z-score of each cell-type score within region) removes the
    between-region differences in marker-gene ranks so that donors sampled in different region sets stay comparable."""
    sc = scores.loc[meta.index].copy()
    if region_center:
        for reg, idx in meta.groupby("region").groups.items():
            sub = sc.loc[idx]
            sc.loc[idx] = (sub - sub.mean()) / sub.std(ddof=0).replace(0, 1.0)
    sc["donor_id"] = meta["donor_id"].values
    return sc.groupby("donor_id").mean()


def run_e2_analysis():
    print("=== RUNNING EXPLORATORY ANALYSIS E2: CELL-TYPE COMPOSITION (DONOR LEVEL) ===")
    os.makedirs("results", exist_ok=True)

    meta_msbb, expr_msbb = load_gse84422()
    cell_scores_msbb = compute_cell_type_scores(expr_msbb)
    cts = list(MARKER_SETS.keys())

    valid_msbb = meta_msbb.dropna(subset=["braak", "donor_id"]).copy()

    # ---- (a) Superseded sample-level analysis, kept for transparency: treats every tissue sample as independent ----
    sample_level = {}
    for ct in cts:
        rho, p_val = calc_spearman(valid_msbb["braak"].values, cell_scores_msbb.loc[valid_msbb.index, ct].values)
        sample_level[ct] = {"spearman_rho": float(rho), "spearman_p": float(p_val), "n": int(len(valid_msbb))}

    # ---- (b) Donor-level analysis (primary): one region-centered profile per donor ----
    donor_meta = valid_msbb.groupby("donor_id").agg(braak=("braak", "first"), age=("age", "first"), pmi=("pmi", "first"),
                                                    sex=("sex", "first"), n_samples=("braak", "size"))
    donor_sc = _donor_level_scores(valid_msbb, cell_scores_msbb)
    donor_sc_raw = _donor_level_scores(valid_msbb, cell_scores_msbb, region_center=False)
    dm = donor_meta.join(donor_sc)
    dm_raw = donor_meta.join(donor_sc_raw)
    n_donors = len(dm)

    rng = np.random.default_rng(42)
    donor_level = {}
    pvals = []
    for ct in cts:
        rho, p_val = calc_spearman(dm["braak"].values, dm[ct].values)
        rho_raw, p_raw = calc_spearman(dm_raw["braak"].values, dm_raw[ct].values)
        boots = []
        for _ in range(2000):
            ix = rng.integers(0, n_donors, n_donors)
            r_b, _ = calc_spearman(dm["braak"].values[ix], dm[ct].values[ix])
            boots.append(r_b)
        donor_level[ct] = {"spearman_rho": float(rho), "spearman_p": float(p_val),
                           "ci_lower": float(np.nanpercentile(boots, 2.5)), "ci_upper": float(np.nanpercentile(boots, 97.5)),
                           "spearman_rho_uncentered": float(rho_raw), "spearman_p_uncentered": float(p_raw)}
        pvals.append(p_val)
        print(f"  {ct:17s} donor-level rho = {rho:+.3f} [{donor_level[ct]['ci_lower']:+.3f}, {donor_level[ct]['ci_upper']:+.3f}] p = {p_val:.3g} (sample-level was {sample_level[ct]['spearman_rho']:+.3f})")
    from statsmodels.stats.multitest import multipletests
    q = multipletests(pvals, method="fdr_bh")[1]
    for ct, qv in zip(cts, q):
        donor_level[ct]["bh_q"] = float(qv)

    # Multivariable donor-level model (same specification as before, now one row per donor)
    dm["is_female"] = (dm["sex"] == "FEMALE").astype(float)
    cols = ["endothelial", "neuronal", "microglial", "astrocytic", "age", "is_female", "pmi"]
    dv = dm.dropna(subset=cols + ["braak"])
    X_mat = sm.add_constant(dv[cols].values, has_constant="add")
    ols_res = sm.OLS(dv["braak"].values, X_mat).fit()
    coef = dict(zip(["const"] + cols, ols_res.params)); pv = dict(zip(["const"] + cols, ols_res.pvalues))
    ci = ols_res.conf_int()
    print(f"\nDonor-level OLS (N={len(dv)}): neuronal beta={coef['neuronal']:+.3f} (p={pv['neuronal']:.3g}); endothelial beta={coef['endothelial']:+.3f} (p={pv['endothelial']:.3g})")

    results = {
        "analysis": "E2_Cell_Type_Composition_DonorLevel",
        "marker_sets": MARKER_SETS,
        "marker_panel_note": "hand-curated 5-8 gene canonical marker panels; NOT the BRETIGEA marker resource",
        "n_donors": int(n_donors),
        "n_samples": int(len(valid_msbb)),
        "donor_level_vs_braak": donor_level,
        "sample_level_vs_braak_superseded": sample_level,
        "multivariable_model": {
            "n_donors": int(len(dv)),
            "endothelial_beta": float(coef["endothelial"]), "endothelial_p_value": float(pv["endothelial"]),
            "endothelial_ci": [float(ci[1][0]), float(ci[1][1])],
            "neuronal_beta": float(coef["neuronal"]), "neuronal_p_value": float(pv["neuronal"]),
            "neuronal_ci": [float(ci[2][0]), float(ci[2][1])],
            "r_squared": float(ols_res.rsquared)
        }
    }

    with open("results/e2_celltype.json", "w") as f:
        json.dump(results, f, indent=2)

    log_analysis(
        analysis_id="E2_CELL_002",
        category="EXPLORATORY",
        hypothesis_or_idea="E2 Cell-Type Marker Scores vs Braak (donor level; supersedes sample-level E2_CELL_001)",
        description="Curated marker-panel rank scores, region-centered and averaged per donor; Spearman vs Braak and donor-level OLS",
        cohorts_used="GSE84422_MSBB",
        regions_analyzed="19 regions (donor mean of region-centered scores)",
        model_type="MarkerPanel_RankScore_DonorLevel_OLS",
        covariates_adjusted="Other cell-type scores, Age, Sex, PMI",
        sample_size_donors=int(len(dv)),
        primary_metric="Neuronal_Spearman_rho_donor_level",
        point_estimate=donor_level["neuronal"]["spearman_rho"],
        ci_95_lower=donor_level["neuronal"]["ci_lower"],
        ci_95_upper=donor_level["neuronal"]["ci_upper"],
        p_value=donor_level["neuronal"]["spearman_p"],
        replicated_in_lockbox=False,
        status="COMPLETE",
        notes=f"Sample-level rho was {sample_level['neuronal']['spearman_rho']:+.3f} (n={len(valid_msbb)} samples treated as independent; superseded)."
    )

    return results


if __name__ == "__main__":
    run_e2_analysis()
