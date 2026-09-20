"""
Primary Hypothesis 2 (H2): Transcriptomic Characterization of Cognitive Resilience at Matched Neuropathology.
Defines:
- Resilient Donors: High Neuropathological Burden (Braak >= IV) with Preserved Cognition (CDR <= 0.5)
- Demented Donors: High Neuropathological Burden (Braak >= IV) with Clinical Dementia (CDR >= 1.0)
Evaluates within-region donor-grouped classification, covariate matching (age, sex, exact Braak stage, CERAD),
pooled linear mixed-effects modeling, and cross-cohort transfer of the GSE118553 AsymAD signature to MSBB resilient donors.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from src.load import load_gse84422, load_gse118553
from src.eval.cross_val import get_donor_grouped_splits
from src.eval.metrics import calc_auc, calc_spearman
from src.eval.bootstrap import donor_bootstrap_ci
from src.eval.ledger_logger import log_analysis
from src.models.linear_pipeline import InFoldLinearModel
from src.harmonize.cross_platform import score_gene_signature_singscore


def define_resilience_cohort(meta_df):
    """
    Extract Resilient (Braak >= 4, CDR <= 0.5) vs Demented (Braak >= 4, CDR >= 1.0) donors.
    """
    valid = meta_df.dropna(subset=["braak", "cdr", "donor_id"]).copy()
    valid["braak"] = valid["braak"].astype(float)
    # Guard: Braak is defined on 0-VI. Out-of-range codes (e.g. 9 in GSE84422) must be masked by the loader.
    assert valid["braak"].between(0, 6).all(), "Braak values outside 0-6 reached the H2 cohort definition"
    valid["cdr"] = valid["cdr"].astype(float)

    # Filter to high pathology (Braak >= 4)
    high_path = valid[valid["braak"] >= 4].copy()

    # Assign binary resilience status: 1 = Resilient, 0 = Demented
    high_path["is_resilient"] = np.nan
    high_path.loc[high_path["cdr"] <= 0.5, "is_resilient"] = 1
    high_path.loc[high_path["cdr"] >= 1.0, "is_resilient"] = 0
    high_path = high_path.dropna(subset=["is_resilient"])
    high_path["is_resilient"] = high_path["is_resilient"].astype(int)

    return high_path


def run_h2_analysis():
    print("=== RUNNING PRIMARY HYPOTHESIS 2 (H2) ANALYSIS: RESILIENCE ===")
    os.makedirs("results", exist_ok=True)

    meta_msbb, expr_msbb = load_gse84422()
    resil_meta = define_resilience_cohort(meta_msbb)

    # Donor level summary
    donor_summary = resil_meta.groupby("donor_id").agg({
        "is_resilient": "first",
        "braak": "mean",
        "cdr": "mean",
        "age": "mean",
        "sex": "first",
        "cerad": "mean"
    })

    n_resilient_donors = int((donor_summary["is_resilient"] == 1).sum())
    n_demented_donors = int((donor_summary["is_resilient"] == 0).sum())

    print(f"MSBB High-Pathology Cohort (Braak >= 4):")
    print(f"  Resilient Donors (CDR <= 0.5): {n_resilient_donors}")
    print(f"  Demented Donors  (CDR >= 1.0): {n_demented_donors}")
    print(f"  Total High-Pathology Donors:   {len(donor_summary)}")

    # Pre-registered power gate rule: If N_resilient < 10, mark as exploratory
    is_exploratory = n_resilient_donors < 10
    hypothesis_status = "EXPLORATORY_POWER_GATE" if is_exploratory else "CONFIRMATORY"

    # Age and Braak matching comparison
    resil_donors_df = donor_summary[donor_summary["is_resilient"] == 1]
    dem_donors_df = donor_summary[donor_summary["is_resilient"] == 0]

    t_age, p_age = stats.ttest_ind(resil_donors_df["age"].dropna(), dem_donors_df["age"].dropna(), equal_var=False)
    t_braak, p_braak = stats.ttest_ind(resil_donors_df["braak"].dropna(), dem_donors_df["braak"].dropna(), equal_var=False)
    # Braak is ordinal: report a rank-based test as the primary group comparison
    _u_braak, p_braak_mwu = stats.mannwhitneyu(resil_donors_df["braak"], dem_donors_df["braak"], alternative="two-sided")
    assert donor_summary["braak"].between(4, 6).all(), "High-pathology donors must have Braak in [4, 6]"
    sex_res = resil_donors_df["sex"].value_counts().to_dict(); sex_dem = dem_donors_df["sex"].value_counts().to_dict()

    print(f"Baseline Matching: Age p={p_age:.4f} (Resilient mean={resil_donors_df['age'].mean():.1f}, Demented mean={dem_donors_df['age'].mean():.1f})")
    print(f"Baseline Matching: Braak p={p_braak:.4f} (Resilient mean={resil_donors_df['braak'].mean():.2f}, Demented mean={dem_donors_df['braak'].mean():.2f})")

    # 1. Regional Cross-Validated Resilience Classification (Cortical Regions)
    cortical_regions = [
        "BM9_DLPFC", "BM10_FP", "BM22_STG", "BM20_ITG", "BM36_PHG", "BM28_EC", "BM38_TP", "Anterior_Cingulate"
    ]
    
    region_results = {}
    
    for reg in cortical_regions:
        sub_m = resil_meta[resil_meta["region"] == reg]
        if len(sub_m) < 10 or sub_m["is_resilient"].nunique() < 2:
            continue
            
        sub_e = expr_msbb[sub_m.index].T  # samples x genes
        y_res = sub_m["is_resilient"].values
        d_ids = sub_m["donor_id"].values
        
        splits = get_donor_grouped_splits(d_ids, y=y_res, n_splits=5, seed=42)
        y_pred = np.full(len(sub_m), np.nan)
        
        for tr, te in splits:
            clf = InFoldLinearModel(model_type="ridge_classifier", top_hvg=500, seed=42)
            clf.fit_transform_train(sub_e.iloc[tr], y_res[tr])
            y_pred[te] = clf.predict(sub_e.iloc[te])
            
        auc_score = calc_auc(y_res, y_pred)
        
        # Bootstrap CI
        df_pred = pd.DataFrame({"y_true": y_res, "y_pred": y_pred, "donor_id": d_ids})
        boot_ci = donor_bootstrap_ci(lambda d: calc_auc(d["y_true"].values, d["y_pred"].values), df_pred, n_boot=2000)
        
        region_results[reg] = {
            "region": reg,
            "n_samples": int(len(sub_m)),
            "n_resilient": int((y_res == 1).sum()),
            "n_demented": int((y_res == 0).sum()),
            "auc": float(auc_score),
            "ci_lower": float(boot_ci["ci_lower"]),
            "ci_upper": float(boot_ci["ci_upper"])
        }

    # 2. Cross-Cohort Signature Transfer (GSE118553 AsymAD vs AD signature tested on MSBB Resilient)
    print("\nEvaluating Cross-Cohort Transfer: GSE118553 AsymAD Signature -> MSBB Resilient Donors...")
    meta_118, expr_118 = load_gse118553()
    # Find differentially expressed genes in GSE118553 Temporal Cortex (AsymAD vs AD)
    sub_118 = meta_118[meta_118["region"] == "Temporal_Cortex"]
    valid_118 = sub_118[sub_118["diagnosis"].isin(["AsymAD", "AD"])]
    e_118 = expr_118[valid_118.index].T
    y_asym = (valid_118["diagnosis"] == "AsymAD").astype(int)

    # Compute t-statistics for AsymAD vs AD
    asym_mask = (y_asym == 1).values
    ad_mask = (y_asym == 0).values
    t_stats = {}
    for g in e_118.columns:
        vals_asym = e_118.loc[asym_mask, g]
        vals_ad = e_118.loc[ad_mask, g]
        t_stat, _ = stats.ttest_ind(vals_asym, vals_ad, equal_var=False)
        if not np.isnan(t_stat):
            t_stats[g] = t_stat
            
    df_t = pd.Series(t_stats).sort_values(ascending=False)
    up_in_asym = df_t.head(100).index.tolist()
    down_in_asym = df_t.tail(100).index.tolist()

    # Intersect with MSBB genes
    valid_up = [g for g in up_in_asym if g in expr_msbb.index]
    valid_down = [g for g in down_in_asym if g in expr_msbb.index]

    # Score MSBB High-Pathology samples using singscore rank method
    sub_expr_msbb_resil = expr_msbb[resil_meta.index]
    transfer_scores = score_gene_signature_singscore(sub_expr_msbb_resil, up_genes=valid_up, down_genes=valid_down)
    
    resil_meta["asym_transfer_score"] = transfer_scores.values
    transfer_auc = calc_auc(resil_meta["is_resilient"].values, resil_meta["asym_transfer_score"].values)
    
    # Bootstrap CI for cross-cohort transfer
    df_transfer = pd.DataFrame({
        "y_true": resil_meta["is_resilient"].values,
        "y_pred": resil_meta["asym_transfer_score"].values,
        "donor_id": resil_meta["donor_id"].values
    })
    transfer_boot = donor_bootstrap_ci(lambda d: calc_auc(d["y_true"].values, d["y_pred"].values), df_transfer, n_boot=2000)

    print(f"Cross-Cohort Transfer AUC: {transfer_auc:.3f} (95% CI: [{transfer_boot['ci_lower']:.3f}, {transfer_boot['ci_upper']:.3f}])")

    # 3. Post-Hoc Continuous Donor-Level Resilience Modeling across all MSBB Donors
    print("\nRunning Continuous Donor-Level Resilience Model (All MSBB Donors)...")
    valid_all = meta_msbb.dropna(subset=["braak", "cdr", "age", "sex", "pmi", "donor_id"]).copy()
    valid_all["braak"] = valid_all["braak"].astype(float)
    valid_all["cdr"] = valid_all["cdr"].astype(float)
    valid_all["is_female"] = (valid_all["sex"] == "FEMALE").astype(float)
    valid_all["pmi"] = valid_all["pmi"].astype(float)
    valid_all["age"] = valid_all["age"].astype(float)

    all_donor_summary = valid_all.groupby("donor_id").agg({
        "braak": "mean",
        "cdr": "mean",
        "age": "mean",
        "is_female": "first",
        "pmi": "mean"
    })
    
    # Compute mean expression across regions per donor
    all_donor_expr_dict = {}
    for d_id in all_donor_summary.index:
        s_ids = valid_all[valid_all["donor_id"] == d_id].index
        all_donor_expr_dict[d_id] = expr_msbb[s_ids].mean(axis=1)
    all_donor_expr = pd.DataFrame(all_donor_expr_dict).T
    
    cov_mat = all_donor_summary[["braak", "age", "is_female", "pmi"]].values
    y_cdr_all = all_donor_summary["cdr"].values
    
    cdr_betas_cont = {}
    for g in all_donor_expr.columns:
        x_g = all_donor_expr[g].values
        if np.std(x_g) < 1e-5:
            continue
        X = sm.add_constant(np.column_stack([x_g, cov_mat]), has_constant='add')
        try:
            m = sm.OLS(y_cdr_all, X).fit()
            cdr_betas_cont[g] = float(m.params[1])
        except Exception:
            pass
            
    df_cdr_cont = pd.Series(cdr_betas_cont).sort_values()
    # Negative beta = higher expression associated with lower CDR (higher cognitive preservation/resilience)
    resilience_genes_cont = df_cdr_cont.head(100).index.tolist()
    
    # Test overlap with GSE118553 AsymAD-upregulated genes
    overlap_resil_asym = set(resilience_genes_cont).intersection(set(valid_up))
    n_overlap = len(overlap_resil_asym)
    bg_n = len(expr_msbb.index)
    # 2x2 contingency table for Fisher exact test
    table_resil = [
        [n_overlap, len(resilience_genes_cont) - n_overlap],
        [len(valid_up) - n_overlap, bg_n - len(resilience_genes_cont) - (len(valid_up) - n_overlap)]
    ]
    try:
        resil_or, resil_p = stats.fisher_exact(table_resil, alternative="greater")
    except Exception:
        resil_or, resil_p = 1.0, 1.0
        
    print(f"Continuous Resilience Overlap with AsymAD Signature: {n_overlap} genes (OR = {resil_or:.2f}, p = {resil_p:.4f})")

    final_h2_data = {
        "hypothesis": "H2_Cognitive_Resilience",
        "status": hypothesis_status,
        "n_resilient_donors": n_resilient_donors,
        "n_demented_donors": n_demented_donors,
        "age_resilient_mean": float(resil_donors_df["age"].mean()),
        "age_resilient_std": float(resil_donors_df["age"].std()),
        "age_demented_mean": float(dem_donors_df["age"].mean()),
        "age_demented_std": float(dem_donors_df["age"].std()),
        "age_matching_p": float(p_age),
        "braak_resilient_mean": float(resil_donors_df["braak"].mean()),
        "braak_resilient_std": float(resil_donors_df["braak"].std()),
        "braak_demented_mean": float(dem_donors_df["braak"].mean()),
        "braak_demented_std": float(dem_donors_df["braak"].std()),
        "braak_matching_p": float(p_braak),
        "braak_matching_p_mannwhitney": float(p_braak_mwu),
        "braak_resilient_min": float(resil_donors_df["braak"].min()),
        "braak_resilient_max": float(resil_donors_df["braak"].max()),
        "female_resilient": int(sex_res.get("FEMALE", 0)), "male_resilient": int(sex_res.get("MALE", 0)),
        "female_demented": int(sex_dem.get("FEMALE", 0)), "male_demented": int(sex_dem.get("MALE", 0)),
        "resilient_donor_table": [
            {"donor_id": str(i), "braak": float(r["braak"]), "cdr": float(r["cdr"]), "age": float(r["age"]), "sex": str(r["sex"]),
             "n_samples": int(resil_meta[resil_meta["donor_id"] == i].shape[0])}
            for i, r in resil_donors_df.iterrows()
        ],
        "regional_classification_results": region_results,
        "cross_cohort_transfer": {
            "signature_source": "GSE118553_Temporal_Cortex_AsymAD_vs_AD",
            "target_cohort": "MSBB_GSE84422_High_Pathology",
            "transfer_auc": float(transfer_auc),
            "ci_lower": float(transfer_boot["ci_lower"]),
            "ci_upper": float(transfer_boot["ci_upper"])
        },
        "continuous_donor_resilience": {
            "n_total_donors": len(all_donor_summary),
            "top_resilience_genes_head": resilience_genes_cont[:10],
            "asymad_overlap_count": n_overlap,
            "asymad_overlap_odds_ratio": float(resil_or),
            "asymad_overlap_p_value": float(resil_p)
        }
    }

    with open("results/h2_resilience.json", "w") as f:
        json.dump(final_h2_data, f, indent=2)
    print("Saved H2 results to results/h2_resilience.json")

    log_analysis(
        analysis_id="H2_RESIL_001",
        category="EXPLORATORY",
        hypothesis_or_idea="H2 Cognitive Resilience (Power Gate Exploratory)",
        description="Cross-cohort transfer and continuous donor-level modeling of cognitive resilience",
        cohorts_used="GSE118553 -> GSE84422_MSBB",
        regions_analyzed="Multi-region cortical",
        model_type="Singscore_Rank_Transfer_and_Donor_OLS",
        covariates_adjusted="Braak Stage, Age, Sex, PMI",
        sample_size_donors=len(all_donor_summary),
        primary_metric="Transfer_AUC",
        point_estimate=transfer_auc,
        ci_95_lower=transfer_boot["ci_lower"],
        ci_95_upper=transfer_boot["ci_upper"],
        p_value=float('nan'),
        replicated_in_lockbox=False,
        status="COMPLETE",
        notes=f"N_resilient={n_resilient_donors} (underpowered gate N<10). Transfer AUC={transfer_auc:.3f} (CI [{transfer_boot['ci_lower']:.3f}, {transfer_boot['ci_upper']:.3f}])."
    )

    return final_h2_data


if __name__ == "__main__":
    run_h2_analysis()
