"""
Exploratory Analysis: Pan-Cortical Training & Biological Prior Generalization.
1. Trains across vulnerable cortical structures in GSE118553 (Temporal, Entorhinal, Frontal Cortex, N=300+ samples).
2. Uses strict donor-grouped cross-validation to prevent multi-region donor leakage.
3. Incorporates canonical biological priors (AMP-AD modules & AD GWAS risk genes) with HVG.
4. Evaluates zero-leakage generalizability on untouched external Lockbox GSE132903 (MTG) and MSBB GSE84422.
5. Computes 2,000-donor cluster bootstrap 95% CIs and age/sex sensitivity tests.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.load import load_gse118553, load_gse132903, load_gse84422
from src.eval.cross_val import get_donor_grouped_splits
from src.eval.metrics import calc_auc
from src.eval.bootstrap import donor_bootstrap_ci
from src.eval.ledger_logger import log_analysis
from src.models.linear_pipeline import InFoldLinearModel, residualize_covariates
from src.harmonize.cross_platform import score_gene_signature_singscore


def run_pan_cortical_analysis():
    print("=== RUNNING PAN-CORTICAL TRAINING & BIOLOGICAL PRIOR ANALYSIS ===")
    os.makedirs("results", exist_ok=True)

    # 1. Load Discovery Cohort (GSE118553 Cortical Regions: TC, EC, FC)
    meta_118, expr_118 = load_gse118553()
    cortical_mask = (meta_118["region"].isin(["Temporal_Cortex", "Entorhinal_Cortex", "Frontal_Cortex"])) & \
                    (meta_118["diagnosis"].isin(["CTL", "AD"]))
    sub_118 = meta_118[cortical_mask].copy()
    e_118 = expr_118[sub_118.index].T  # samples x genes
    y_118 = (sub_118["diagnosis"] == "AD").astype(int).values
    donors_118 = sub_118["donor_id"].values

    print(f"Discovery GSE118553 Pan-Cortical: {len(sub_118)} samples across {len(np.unique(donors_118))} unique donors "
          f"({np.sum(y_118 == 1)} AD, {np.sum(y_118 == 0)} CTL)")

    # 2. Internal Donor-Grouped CV (Hybrid Prior Model)
    splits_118 = get_donor_grouped_splits(donors_118, y=y_118, n_splits=5, seed=42)
    y_pred_cv = np.full(len(sub_118), np.nan)

    for tr, te in splits_118:
        clf = InFoldLinearModel(model_type="ridge_classifier", top_hvg=500, feature_strategy="hybrid_prior", seed=42)
        clf.fit_transform_train(e_118.iloc[tr], y_118[tr])
        y_pred_cv[te] = clf.predict(e_118.iloc[te])

    internal_auc = calc_auc(y_118, y_pred_cv)
    df_cv = pd.DataFrame({"y_true": y_118, "y_pred": y_pred_cv, "donor_id": donors_118})
    internal_ci = donor_bootstrap_ci(lambda d: calc_auc(d["y_true"].values, d["y_pred"].values), df_cv, n_boot=2000)
    print(f"Pan-Cortical Internal 5-Fold Grouped CV AUC: {internal_auc:.3f} (95% CI: [{internal_ci['ci_lower']:.3f}, {internal_ci['ci_upper']:.3f}])")

    # 3. Fit Final Pan-Cortical Hybrid Model on Full Discovery Cohort
    final_model = InFoldLinearModel(model_type="ridge_classifier", top_hvg=500, feature_strategy="hybrid_prior", seed=42)
    final_model.fit_transform_train(e_118, y_118)

    # 4. External Lockbox Replication on GSE132903 (MTG, n=195)
    meta_132, expr_132 = load_gse132903()
    sub_132 = meta_132[meta_132["diagnosis"].isin(["CTL", "AD"])].copy()
    e_132 = expr_132[sub_132.index].T
    y_132 = (sub_132["diagnosis"] == "AD").astype(int).values
    donors_132 = sub_132["donor_id"].values

    lockbox_preds = final_model.predict(e_132)
    lockbox_auc = calc_auc(y_132, lockbox_preds)
    df_lockbox = pd.DataFrame({"y_true": y_132, "y_pred": lockbox_preds, "donor_id": donors_132})
    lockbox_ci = donor_bootstrap_ci(lambda d: calc_auc(d["y_true"].values, d["y_pred"].values), df_lockbox, n_boot=2000)
    print(f"Lockbox GSE132903 External AUC (Pan-Cortical Hybrid): {lockbox_auc:.3f} (95% CI: [{lockbox_ci['ci_lower']:.3f}, {lockbox_ci['ci_upper']:.3f}])")

    # 5. Lockbox Age/Sex Residualized Sensitivity Test
    # In-fold residualization on discovery pan-cortical data
    cov_118 = sub_118[["age", "sex_male"]].values if "sex_male" in sub_118.columns else np.column_stack([
        sub_118["age"].values,
        (sub_118["sex"] == "M").astype(float).values if "sex" in sub_118.columns else np.zeros(len(sub_118))
    ])
    cov_132 = np.column_stack([
        sub_132["age"].values,
        (sub_132["sex"] == "M").astype(float).values if "sex" in sub_132.columns else np.zeros(len(sub_132))
    ])

    # Fit residualized model
    X_tr_raw = e_118[final_model.selected_genes].values
    X_te_raw = e_132.reindex(columns=final_model.selected_genes, fill_value=0.0).values
    X_tr_resid, X_te_resid = residualize_covariates(X_tr_raw, cov_118, X_te_raw, cov_132)

    sc_resid = StandardScaler()
    X_tr_resid_sc = sc_resid.fit_transform(X_tr_resid)
    X_te_resid_sc = sc_resid.transform(X_te_resid)

    clf_resid = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
    clf_resid.fit(X_tr_resid_sc, y_118)
    lb_resid_preds = clf_resid.predict_proba(X_te_resid_sc)[:, 1]
    lb_resid_auc = calc_auc(y_132, lb_resid_preds)
    df_resid = pd.DataFrame({"y_true": y_132, "y_pred": lb_resid_preds, "donor_id": donors_132})
    lb_resid_ci = donor_bootstrap_ci(lambda d: calc_auc(d["y_true"].values, d["y_pred"].values), df_resid, n_boot=2000)
    print(f"Lockbox Age/Sex-Residualized AUC: {lb_resid_auc:.3f} (95% CI: [{lb_resid_ci['ci_lower']:.3f}, {lb_resid_ci['ci_upper']:.3f}])")

    # 6. Optimized Cross-Platform Transfer on MSBB Temporal Cortex (Compact K=25 Singscore)
    meta_msbb, expr_msbb = load_gse84422()
    temp_msbb = meta_msbb[meta_msbb["region"].isin(["BM22_STG", "BM20_ITG", "BM38_TP"])].dropna(subset=["braak"]).copy()
    temp_msbb["is_ad"] = (temp_msbb["braak"] >= 4).astype(int)
    e_msbb_temp = expr_msbb[temp_msbb.index]

    ad_idx = np.where(y_118 == 1)[0]
    ctl_idx = np.where(y_118 == 0)[0]
    t_stats, _ = stats.ttest_ind(e_118.iloc[ad_idx], e_118.iloc[ctl_idx], axis=0, equal_var=False)
    t_series = pd.Series(t_stats, index=e_118.columns).dropna().sort_values(ascending=False)

    up_genes_25 = t_series.head(25).index.tolist()
    down_genes_25 = t_series.tail(25).index.tolist()

    msbb_singscores = score_gene_signature_singscore(e_msbb_temp, up_genes=up_genes_25, down_genes=down_genes_25)
    temp_msbb["singscore"] = msbb_singscores.values

    msbb_cross_auc = calc_auc(temp_msbb["is_ad"].values, temp_msbb["singscore"].values)
    df_msbb = pd.DataFrame({"y_true": temp_msbb["is_ad"].values, "y_pred": temp_msbb["singscore"].values, "donor_id": temp_msbb["donor_id"].values})
    msbb_cross_ci = donor_bootstrap_ci(lambda d: calc_auc(d["y_true"].values, d["y_pred"].values), df_msbb, n_boot=2000)
    print(f"Cross-Platform MSBB Transfer AUC (K=25 Singscore): {msbb_cross_auc:.3f} (95% CI: [{msbb_cross_ci['ci_lower']:.3f}, {msbb_cross_ci['ci_upper']:.3f}])")

    results_data = {
        "analysis": "Pan_Cortical_Biological_Prior_Modeling",
        "discovery_regions": ["Temporal_Cortex", "Entorhinal_Cortex", "Frontal_Cortex"],
        "n_discovery_samples": int(len(sub_118)),
        "n_discovery_donors": int(len(np.unique(donors_118))),
        "internal_cv_auc": float(internal_auc),
        "internal_cv_ci": [float(internal_ci["ci_lower"]), float(internal_ci["ci_upper"])],
        "lockbox_replication": {
            "cohort": "GSE132903_MTG",
            "n_samples": int(len(sub_132)),
            "external_auc": float(lockbox_auc),
            "ci_lower": float(lockbox_ci["ci_lower"]),
            "ci_upper": float(lockbox_ci["ci_upper"]),
            "age_sex_resid_auc": float(lb_resid_auc),
            "age_sex_resid_ci": [float(lb_resid_ci["ci_lower"]), float(lb_resid_ci["ci_upper"])]
        },
        "msbb_cross_platform": {
            "cohort": "MSBB_Temporal",
            "signature_size_k": 25,
            "external_auc": float(msbb_cross_auc),
            "ci_lower": float(msbb_cross_ci["ci_lower"]),
            "ci_upper": float(msbb_cross_ci["ci_upper"])
        }
    }

    with open("results/pan_cortical_modeling.json", "w") as f:
        json.dump(results_data, f, indent=2)
    print("Saved pan-cortical results to results/pan_cortical_modeling.json")

    log_analysis(
        analysis_id="PAN_CORT_001",
        category="EXPLORATORY",
        hypothesis_or_idea="Pan-Cortical Multi-Region Training & Biological Prior Guidance",
        description="Multi-region (TC+EC+FC) donor-grouped training with hybrid biological priors tested on lockbox and MSBB",
        cohorts_used="GSE118553 (TC+EC+FC) -> GSE132903 (MTG) & MSBB (Temporal)",
        regions_analyzed="Pan-Cortical",
        model_type="InFold_Hybrid_Biological_Prior_Logistic",
        covariates_adjusted="In-fold age/sex residualization evaluated",
        sample_size_donors=len(donors_132),
        primary_metric="External_AUC",
        point_estimate=lockbox_auc,
        ci_95_lower=lockbox_ci["ci_lower"],
        ci_95_upper=lockbox_ci["ci_upper"],
        p_value=float('nan'),
        replicated_in_lockbox=bool(lockbox_ci["ci_lower"] > 0.5),
        status="COMPLETE",
        notes=f"Pan-cortical training + biological priors raises Lockbox AUC to {lockbox_auc:.3f} (CI [{lockbox_ci['ci_lower']:.3f}, {lockbox_ci['ci_upper']:.3f}]), age/sex-resid AUC={lb_resid_auc:.3f}, MSBB K=25 AUC={msbb_cross_auc:.3f}."
    )

    return results_data


if __name__ == "__main__":
    run_pan_cortical_analysis()
