"""
Exploratory Analysis E1: Leakage Taxonomy Benchmark.
Empirically quantifies optimism bias and AUC inflation across four distinct data leakage modes:
1. Gold Standard (Zero Leakage): Donor-grouped CV, in-fold feature selection, in-fold scaling, in-fold tuning
2. Donor Leakage: Random sample CV splitting samples from the same donor across train and test folds
3. Feature Selection Leakage: Selecting top HVGs on the entire dataset prior to splitting
4. Normalization Leakage: Global z-scoring across all samples before fold partitioning
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.load import load_gse118553
from src.eval.cross_val import get_donor_grouped_splits
from src.eval.metrics import calc_auc
from src.eval.bootstrap import donor_bootstrap_ci
from src.eval.ledger_logger import log_analysis
from src.models.linear_pipeline import InFoldLinearModel


def run_e1_analysis():
    print("=== RUNNING EXPLORATORY ANALYSIS E1: LEAKAGE TAXONOMY ===")
    os.makedirs("results", exist_ok=True)

    meta, expr = load_gse118553()
    # Filter to AD vs CTL across all regions
    valid = meta[meta["diagnosis"].isin(["CTL", "AD"])].copy()
    X_df = expr[valid.index].T  # samples x genes
    y = (valid["diagnosis"] == "AD").astype(int).values
    donors = valid["donor_id"].values
    n_samples = len(valid)

    print(f"Cohort for Leakage Benchmark: {n_samples} samples across {len(np.unique(donors))} donors.")

    # 1. Gold Standard: Zero Leakage
    splits_grouped = get_donor_grouped_splits(donors, y=y, n_splits=5, seed=42)
    y_pred_gold = np.full(n_samples, np.nan)
    for tr, te in splits_grouped:
        model = InFoldLinearModel(model_type="ridge_classifier", top_hvg=500, seed=42)
        model.fit_transform_train(X_df.iloc[tr], y[tr])
        y_pred_gold[te] = model.predict(X_df.iloc[te])
    auc_gold = calc_auc(y, y_pred_gold)

    # 1b. Matched-classifier reference: identical to the leaky arms below (plain L2 logistic regression, C=1) but with
    # donor-grouped splits and ALL preprocessing (HVG selection, z-scoring) fitted on the training fold only.
    # Leakage effects are measured against THIS reference so that the classifier is not confounded with the leakage mode.
    y_pred_ref = np.full(n_samples, np.nan)
    for tr, te in splits_grouped:
        hv = X_df.iloc[tr].var(axis=0).nlargest(500).index
        sc = StandardScaler().fit(X_df.iloc[tr][hv].values)
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(sc.transform(X_df.iloc[tr][hv].values), y[tr])
        y_pred_ref[te] = clf.predict_proba(sc.transform(X_df.iloc[te][hv].values))[:, 1]
    auc_ref = calc_auc(y, y_pred_ref)

    # 2. Donor Leakage: Random sample K-Fold (ignoring donor clusters)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_donor_leak = np.full(n_samples, np.nan)
    for tr, te in kf.split(X_df):
        hv = X_df.iloc[tr].var(axis=0).nlargest(500).index
        sc = StandardScaler().fit(X_df.iloc[tr][hv].values)
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(sc.transform(X_df.iloc[tr][hv].values), y[tr])
        y_pred_donor_leak[te] = clf.predict_proba(sc.transform(X_df.iloc[te][hv].values))[:, 1]
    auc_donor_leak = calc_auc(y, y_pred_donor_leak)

    # 3. Feature Selection Leakage: HVG selection on pooled dataset before splitting
    variances = X_df.var(axis=0)
    leaked_hvg = variances.nlargest(500).index
    X_leaked_feat = X_df[leaked_hvg]
    y_pred_feat_leak = np.full(n_samples, np.nan)
    for tr, te in splits_grouped:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_leaked_feat.iloc[tr].values)
        X_te = scaler.transform(X_leaked_feat.iloc[te].values)
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X_tr, y[tr])
        y_pred_feat_leak[te] = clf.predict_proba(X_te)[:, 1]
    auc_feat_leak = calc_auc(y, y_pred_feat_leak)

    # 4. Normalization Leakage: Global z-score before splitting
    scaler_global = StandardScaler()
    X_global_scaled = pd.DataFrame(scaler_global.fit_transform(X_df.values), index=X_df.index, columns=X_df.columns)
    y_pred_norm_leak = np.full(n_samples, np.nan)
    for tr, te in splits_grouped:
        # genes chosen by RAW in-fold variance (z-scoring equalises variances, so selecting on the scaled matrix would be arbitrary);
        # the only leak in this arm is that the scaling parameters were estimated on all samples, test fold included
        hvg_tr = X_df.iloc[tr].var(axis=0).nlargest(500).index
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X_global_scaled.iloc[tr][hvg_tr].values, y[tr])
        y_pred_norm_leak[te] = clf.predict_proba(X_global_scaled.iloc[te][hvg_tr].values)[:, 1]
    auc_norm_leak = calc_auc(y, y_pred_norm_leak)

    # 5. Compound Leakage (Donor + Feature + Normalization Leakage)
    y_pred_compound = np.full(n_samples, np.nan)
    X_compound = pd.DataFrame(scaler_global.fit_transform(X_df[leaked_hvg].values), index=X_df.index, columns=leaked_hvg)
    for tr, te in kf.split(X_compound):
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X_compound.iloc[tr].values, y[tr])
        y_pred_compound[te] = clf.predict_proba(X_compound.iloc[te].values)[:, 1]
    auc_compound = calc_auc(y, y_pred_compound)

    results = {
        "analysis": "E1_Leakage_Taxonomy",
        "design_note": "All arms use the same classifier (L2 logistic, C=1) on 500 top-variance genes; only the leakage mode differs. Pooled four GSE118553 regions, AD vs CTL. Draft 8 compared leaky arms (C=1) with a CV-tuned reference, which confounded classifier with leakage.",
        "n_samples": int(n_samples), "n_donors": int(len(np.unique(donors))),
        "gold_standard_zero_leakage_auc": float(auc_ref),
        "tuned_classifier_zero_leakage_auc": float(auc_gold),
        "donor_leakage_auc": float(auc_donor_leak),
        "donor_leakage_inflation": float(auc_donor_leak - auc_ref),
        "feature_selection_leakage_auc": float(auc_feat_leak),
        "feature_selection_inflation": float(auc_feat_leak - auc_ref),
        "normalization_leakage_auc": float(auc_norm_leak),
        "normalization_inflation": float(auc_norm_leak - auc_ref),
        "compound_leakage_auc": float(auc_compound),
        "compound_inflation": float(auc_compound - auc_ref)
    }

    print("\nLeakage Benchmark Results:")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    with open("results/e1_leakage.json", "w") as f:
        json.dump(results, f, indent=2)

    log_analysis(
        analysis_id="E1_LEAK_001",
        category="EXPLORATORY",
        hypothesis_or_idea="E1 Leakage Taxonomy Benchmark",
        description="Empirical quantification of optimism bias across donor, feature, normalization, and compound leakage modes",
        cohorts_used="GSE118553",
        regions_analyzed="Multi-region",
        model_type="Logistic_L2_Comparison",
        covariates_adjusted="None",
        sample_size_donors=len(np.unique(donors)),
        primary_metric="AUC_Inflation_Compound",
        point_estimate=auc_compound - auc_ref,
        ci_95_lower=float('nan'),
        ci_95_upper=float('nan'),
        p_value=float('nan'),
        replicated_in_lockbox=False,
        status="COMPLETE",
        notes=f"Matched reference AUC={auc_ref:.3f}, compound leakage AUC={auc_compound:.3f} (+{auc_compound - auc_ref:.3f}); tuned-classifier reference {auc_gold:.3f}."
    )

    return results


if __name__ == "__main__":
    run_e1_analysis()
