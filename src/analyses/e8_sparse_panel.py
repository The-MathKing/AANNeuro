"""
Exploratory Analysis E8: Sparse Stability-Selected Biomarker Panel (<= 20 Genes).
1. Fits Elastic Net with Stability Selection across 100 donor-subsampled iterations in GSE118553.
2. Isolates a parsimonious signature panel of <= 20 highly stable transcriptomic features.
3. Tests external zero-leakage generalizability on untouched Lockbox GSE132903.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.load import load_gse118553, load_gse132903
from src.eval.cross_val import get_donor_grouped_splits
from src.eval.metrics import calc_auc
from src.eval.bootstrap import donor_bootstrap_ci
from src.eval.ledger_logger import log_analysis
from src.models.sparse_panel import stability_selection_panel


def run_e8_analysis():
    print("=== RUNNING EXPLORATORY ANALYSIS E8: SPARSE BIOMARKER PANEL ===")
    os.makedirs("results", exist_ok=True)

    # 1. Load Discovery Cohort (GSE118553 Temporal Cortex)
    meta_118, expr_118 = load_gse118553()
    sub_118 = meta_118[(meta_118["region"] == "Temporal_Cortex") & (meta_118["diagnosis"].isin(["CTL", "AD"]))].copy()
    X_118 = expr_118[sub_118.index].T
    y_118 = (sub_118["diagnosis"] == "AD").astype(int)
    donors_118 = sub_118["donor_id"]

    # Pre-filter top 1000 HVGs
    variances = X_118.var(axis=0)
    top_1000_genes = variances.nlargest(1000).index
    X_sub = X_118[top_1000_genes]

    print(f"Running Stability Selection on {len(sub_118)} samples...")
    panel_df = stability_selection_panel(
        X_sub, y_118, donors_118, n_subsamples=100, max_genes=20, l1_ratio=0.7, C=0.5, seed=42
    )

    selected_genes = panel_df["gene"].tolist()
    print(f"\nStability-Selected Panel ({len(selected_genes)} Genes):")
    print(panel_df.to_string(index=False))

    # 2. Internal Nested Grouped CV on Discovery (In-Fold Stability Selection to Prevent Leakage)
    splits_118 = get_donor_grouped_splits(donors_118.values, y=y_118.values, n_splits=5, seed=42)
    y_pred_cv = np.full(len(sub_118), np.nan)
    
    for tr, te in splits_118:
        X_tr_full = X_118.iloc[tr]
        y_tr = y_118.iloc[tr]
        donors_tr = donors_118.iloc[tr]
        
        # In-fold top HVGs
        hvg_infold = X_tr_full.var(axis=0).nlargest(500).index
        X_tr_sub = X_tr_full[hvg_infold]
        
        # In-fold stability selection (30 subsamples for rapid CV)
        fold_panel = stability_selection_panel(
            X_tr_sub, y_tr, donors_tr, n_subsamples=30, max_genes=20, l1_ratio=0.7, C=0.5, seed=42
        )
        fold_genes = fold_panel["gene"].tolist()
        
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_full[fold_genes].values)
        X_te_scaled = scaler.transform(X_118.iloc[te][fold_genes].values)
        
        clf = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)
        clf.fit(X_tr_scaled, y_tr.values)
        y_pred_cv[te] = clf.predict_proba(X_te_scaled)[:, 1]

    internal_sparse_auc = calc_auc(y_118.values, y_pred_cv)
    print(f"\nInternal 5-Fold Nested Grouped CV AUC (Strict In-Fold Panel Selection): {internal_sparse_auc:.3f}")

    # 3. Fit Model on All Discovery Samples
    scaler_final = StandardScaler()
    X_tr_final = scaler_final.fit_transform(X_118[selected_genes].values)
    final_clf = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)
    final_clf.fit(X_tr_final, y_118.values)

    # 4. External Lockbox Replication on GSE132903
    meta_132, expr_132 = load_gse132903()
    sub_132 = meta_132[meta_132["diagnosis"].isin(["CTL", "AD"])].copy()
    y_132 = (sub_132["diagnosis"] == "AD").astype(int).values
    donors_132 = sub_132["donor_id"].values
    
    # Intersect panel genes
    avail_genes = [g for g in selected_genes if g in expr_132.index]
    print(f"Lockbox Panel Gene Overlap: {len(avail_genes)} / {len(selected_genes)}")
    
    X_132 = expr_132.reindex(index=selected_genes, fill_value=0.0)[sub_132.index].T.fillna(0.0).values
    X_132_scaled = scaler_final.transform(X_132)
    lockbox_preds = final_clf.predict_proba(X_132_scaled)[:, 1]
    
    lockbox_sparse_auc = calc_auc(y_132, lockbox_preds)
    df_lockbox = pd.DataFrame({"y_true": y_132, "y_pred": lockbox_preds, "donor_id": donors_132})
    boot_ci = donor_bootstrap_ci(lambda d: calc_auc(d["y_true"].values, d["y_pred"].values), df_lockbox, n_boot=2000)

    print(f"Lockbox External AUC (20-Gene Panel): {lockbox_sparse_auc:.3f} (95% CI: [{boot_ci['ci_lower']:.3f}, {boot_ci['ci_upper']:.3f}])")

    results = {
        "analysis": "E8_Sparse_Signature_Panel",
        "panel_size": len(selected_genes),
        "selected_genes": panel_df.to_dict(orient="records"),
        "internal_cv_auc": float(internal_sparse_auc),
        "external_lockbox_auc": float(lockbox_sparse_auc),
        "external_ci_lower": float(boot_ci["ci_lower"]),
        "external_ci_upper": float(boot_ci["ci_upper"]),
        "replicated": bool(boot_ci["ci_lower"] > 0.50)
    }

    with open("results/e8_sparse_panel.json", "w") as f:
        json.dump(results, f, indent=2)

    log_analysis(
        analysis_id="E8_PANEL_001",
        category="EXPLORATORY",
        hypothesis_or_idea="E8 Sparse Stability-Selected Panel",
        description="20-gene stability-selected elastic net panel validated on external lockbox cohort",
        cohorts_used="GSE118553 -> GSE132903",
        regions_analyzed="Temporal Cortex",
        model_type="ElasticNet_Stability_Selection",
        covariates_adjusted="None",
        sample_size_donors=len(donors_132),
        primary_metric="Lockbox_Sparse_AUC",
        point_estimate=lockbox_sparse_auc,
        ci_95_lower=boot_ci["ci_lower"],
        ci_95_upper=boot_ci["ci_upper"],
        p_value=float('nan'),
        replicated_in_lockbox=bool(boot_ci["ci_lower"] > 0.5),
        status="COMPLETE",
        notes=f"20-gene panel internal AUC={internal_sparse_auc:.3f} -> Lockbox external AUC={lockbox_sparse_auc:.3f} (CI [{boot_ci['ci_lower']:.3f}, {boot_ci['ci_upper']:.3f}])."
    )

    return results


if __name__ == "__main__":
    run_e8_analysis()
