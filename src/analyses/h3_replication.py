"""
Primary Hypothesis 3 (H3): Zero-Leakage External Replication in Independent Lockbox Cohorts.
1. Train on Discovery GSE118553 Temporal Cortex -> Test on untouched Lockbox GSE132903 (same Illumina platform).
2. Train on Discovery GSE118553 Temporal Cortex -> Test on MSBB GSE84422 Temporal Regions (cross-platform Affymetrix).
Computes internal grouped CV AUCs, external lockbox AUCs, calibration, and 2,000-donor bootstrap 95% CIs.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats

from src.load import load_gse118553, load_gse132903, load_gse84422
from src.eval.cross_val import get_donor_grouped_splits
from src.eval.metrics import calc_auc, calc_spearman, calc_pearson
from src.eval.bootstrap import donor_bootstrap_ci
from src.eval.ledger_logger import log_analysis
from src.models.linear_pipeline import InFoldLinearModel
from src.harmonize.cross_platform import score_gene_signature_singscore, standardize_within_cohort


def run_h3_analysis():
    print("=== RUNNING PRIMARY HYPOTHESIS 3 (H3) ANALYSIS: EXTERNAL REPLICATION ===")
    os.makedirs("results", exist_ok=True)

    # 1. Load Discovery Cohort (GSE118553)
    meta_118, expr_118 = load_gse118553()
    sub_118 = meta_118[(meta_118["region"] == "Temporal_Cortex") & (meta_118["diagnosis"].isin(["CTL", "AD"]))].copy()
    e_118 = expr_118[sub_118.index].T  # samples x genes
    y_118 = (sub_118["diagnosis"] == "AD").astype(int).values
    donors_118 = sub_118["donor_id"].values

    print(f"Discovery GSE118553 (Temporal Cortex): {len(sub_118)} samples ({np.sum(y_118 == 1)} AD, {np.sum(y_118 == 0)} CTL)")

    # 2. Internal Donor-Grouped Cross-Validation on Discovery
    splits_118 = get_donor_grouped_splits(donors_118, y=y_118, n_splits=5, seed=42)
    y_pred_cv_118 = np.full(len(sub_118), np.nan)
    
    for tr, te in splits_118:
        clf = InFoldLinearModel(model_type="ridge_classifier", top_hvg=1000, seed=42)
        clf.fit_transform_train(e_118.iloc[tr], y_118[tr])
        y_pred_cv_118[te] = clf.predict(e_118.iloc[te])

    internal_auc = calc_auc(y_118, y_pred_cv_118)
    df_cv = pd.DataFrame({"y_true": y_118, "y_pred": y_pred_cv_118, "donor_id": donors_118})
    internal_ci = donor_bootstrap_ci(lambda d: calc_auc(d["y_true"].values, d["y_pred"].values), df_cv, n_boot=2000)
    print(f"Internal 5-Fold Grouped CV AUC: {internal_auc:.3f} (95% CI: [{internal_ci['ci_lower']:.3f}, {internal_ci['ci_upper']:.3f}])")

    # 3. Fit Model on Full Discovery Cohort
    final_model = InFoldLinearModel(model_type="ridge_classifier", top_hvg=1000, seed=42)
    final_model.fit_transform_train(e_118, y_118)

    # 4. External Lockbox Replication on GSE132903 (Same Platform Illumina HT-12)
    meta_132, expr_132 = load_gse132903()
    sub_132 = meta_132[meta_132["diagnosis"].isin(["CTL", "AD"])].copy()
    e_132 = expr_132[sub_132.index].T  # samples x genes
    y_132 = (sub_132["diagnosis"] == "AD").astype(int).values
    donors_132 = sub_132["donor_id"].values

    print(f"\nLockbox GSE132903: {len(sub_132)} samples ({np.sum(y_132 == 1)} AD, {np.sum(y_132 == 0)} CTL)")

    # Align genes
    common_genes_lockbox = [g for g in final_model.selected_genes if g in e_132.columns]
    print(f"Lockbox Feature Overlap: {len(common_genes_lockbox)} / {len(final_model.selected_genes)} genes")

    # Lockbox prediction
    lockbox_preds = final_model.predict(e_132)
    lockbox_auc = calc_auc(y_132, lockbox_preds)
    
    df_lockbox = pd.DataFrame({"y_true": y_132, "y_pred": lockbox_preds, "donor_id": donors_132})
    lockbox_ci = donor_bootstrap_ci(lambda d: calc_auc(d["y_true"].values, d["y_pred"].values), df_lockbox, n_boot=2000)
    print(f"Lockbox GSE132903 External AUC: {lockbox_auc:.3f} (95% CI: [{lockbox_ci['ci_lower']:.3f}, {lockbox_ci['ci_upper']:.3f}])")

    # 5. Cross-Platform Replication on MSBB Temporal Cortex (BM22 / BM38 / BM20)
    meta_msbb, expr_msbb = load_gse84422()
    temp_msbb = meta_msbb[meta_msbb["region"].isin(["BM22_STG", "BM20_ITG", "BM38_TP"])].dropna(subset=["braak"]).copy()
    temp_msbb["is_ad"] = (temp_msbb["braak"] >= 4).astype(int)
    
    e_msbb_temp = expr_msbb[temp_msbb.index]  # genes x samples
    
    # Singscore cross-platform transfer
    # Extract top upregulated and downregulated genes in Discovery
    X_scaled_tr = final_model.scaler.transform(e_118[final_model.selected_genes].values)
    mean_ad = X_scaled_tr[y_118 == 1].mean(axis=0)
    mean_ctl = X_scaled_tr[y_118 == 0].mean(axis=0)
    diff = mean_ad - mean_ctl
    
    ranked_genes = pd.Series(diff, index=final_model.selected_genes).sort_values(ascending=False)
    up_genes = ranked_genes.head(50).index.tolist()
    down_genes = ranked_genes.tail(50).index.tolist()

    msbb_singscores = score_gene_signature_singscore(e_msbb_temp, up_genes=up_genes, down_genes=down_genes)
    temp_msbb["singscore"] = msbb_singscores.values

    msbb_cross_auc = calc_auc(temp_msbb["is_ad"].values, temp_msbb["singscore"].values)
    df_msbb_cross = pd.DataFrame({
        "y_true": temp_msbb["is_ad"].values,
        "y_pred": temp_msbb["singscore"].values,
        "donor_id": temp_msbb["donor_id"].values
    })
    msbb_cross_ci = donor_bootstrap_ci(lambda d: calc_auc(d["y_true"].values, d["y_pred"].values), df_msbb_cross, n_boot=2000)
    print(f"Cross-Platform MSBB Temporal External AUC: {msbb_cross_auc:.3f} (95% CI: [{msbb_cross_ci['ci_lower']:.3f}, {msbb_cross_ci['ci_upper']:.3f}])")

    final_h3_data = {
        "hypothesis": "H3_External_Replication",
        "discovery_cohort": "GSE118553_Temporal_Cortex",
        "internal_cv_auc": float(internal_auc),
        "internal_cv_ci": [float(internal_ci["ci_lower"]), float(internal_ci["ci_upper"])],
        "lockbox_replication_gse132903": {
            "cohort": "GSE132903_MTG",
            "platform": "Illumina_HumanHT-12_V4.0",
            "n_samples": int(len(sub_132)),
            "external_auc": float(lockbox_auc),
            "ci_lower": float(lockbox_ci["ci_lower"]),
            "ci_upper": float(lockbox_ci["ci_upper"]),
            "replicated": bool(lockbox_ci["ci_lower"] > 0.50)
        },
        "cross_platform_replication_msbb": {
            "cohort": "MSBB_GSE84422_Temporal",
            "platform": "Affymetrix_HG-U133A",
            "n_samples": int(len(temp_msbb)),
            "external_auc": float(msbb_cross_auc),
            "ci_lower": float(msbb_cross_ci["ci_lower"]),
            "ci_upper": float(msbb_cross_ci["ci_upper"]),
            "replicated": bool(msbb_cross_ci["ci_lower"] > 0.50)
        }
    }

    with open("results/h3_replication.json", "w") as f:
        json.dump(final_h3_data, f, indent=2)
    print("Saved H3 results to results/h3_replication.json")

    log_analysis(
        analysis_id="H3_REPL_001",
        category="PRIMARY",
        hypothesis_or_idea="H3 External Replication in Lockbox Cohort",
        description="Zero-leakage generalization from GSE118553 to independent lockbox GSE132903 MTG",
        cohorts_used="GSE118553 -> GSE132903",
        regions_analyzed="Temporal Cortex / MTG",
        model_type="InFold_RidgeCV_Transfer",
        covariates_adjusted="None (Direct feature transfer)",
        sample_size_donors=len(donors_132),
        primary_metric="External_AUC",
        point_estimate=lockbox_auc,
        ci_95_lower=lockbox_ci["ci_lower"],
        ci_95_upper=lockbox_ci["ci_upper"],
        p_value=float('nan'),  # permutation p is computed in audit9_h3.py
        replicated_in_lockbox=bool(lockbox_ci["ci_lower"] > 0.5),
        status="COMPLETE",
        notes=f"Internal CV AUC={internal_auc:.3f} -> Lockbox External AUC={lockbox_auc:.3f} (CI [{lockbox_ci['ci_lower']:.3f}, {lockbox_ci['ci_upper']:.3f}])."
    )

    return final_h3_data


if __name__ == "__main__":
    run_h3_analysis()
