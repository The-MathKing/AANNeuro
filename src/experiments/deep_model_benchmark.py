"""
Deep Benchmarking Engine: Rigorous statistical validation of model improvement architectures.

Evaluates:
  1. Baseline H3 (Temporal Cortex, 1000 HVG, L2 Logistic)
  2. Curated Biological Priors (~70 genes)
  3. Hybrid Priors (500 HVG + Biological Priors)
  4. Within-Sample Relative Rank Transform
  5. Multi-Region Pooled Cortical Training (TC+EC+FC, N=300+)
  6. Composite Model A: Multi-Region + Hybrid Priors
  7. Composite Model B: Multi-Region + Rank Transform
  8. Composite Model C: Multi-Region + Hybrid Priors + Rank Transform (Full Synergy)

For each model:
  - 5-Fold Donor-Grouped Internal CV AUC
  - External Lockbox AUC on GSE132903 (N=195)
  - Donor-Cluster Bootstrap (B=2,000) 95% Confidence Intervals
  - Label Permutation Test (B=1,000)
  - Age & Sex In-Fold Residualized Lockbox AUC
  - Feature Coefficient Extraction for Pathway Attribution
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.abspath("."))

from src.load import load_gse118553, load_gse132903, load_gse84422
from src.eval.cross_val import get_donor_grouped_splits
from src.eval.metrics import calc_auc
from src.eval.bootstrap import donor_bootstrap_ci
from src.eval.permutation import donor_permutation_test
from src.models.linear_pipeline import CANONICAL_BIOLOGICAL_PRIORS, residualize_covariates
from src.harmonize.cross_platform import score_gene_signature_singscore


def run_deep_benchmark():
    print("\n" + "=" * 75)
    print("STARTING DEEP BENCHMARKING OF MODEL IMPROVEMENT ARCHITECTURES")
    print("=" * 75)

    print("Loading transcriptomic cohorts...")
    meta_118, expr_118 = load_gse118553()
    meta_132, expr_132 = load_gse132903()
    meta_msbb, expr_msbb = load_gse84422()

    # 1. Base GSE118553 Temporal Cortex
    sub_118_tc = meta_118[(meta_118["region"] == "Temporal_Cortex") & (meta_118["diagnosis"].isin(["CTL", "AD"]))].copy()
    sub_118_tc["is_female"] = (sub_118_tc["sex"] == "FEMALE").astype(float)
    e_118_tc = expr_118[sub_118_tc.index].T
    y_118_tc = (sub_118_tc["diagnosis"] == "AD").astype(int).values
    donors_118_tc = sub_118_tc["donor_id"].values
    cov_118_tc = sub_118_tc[["age", "is_female"]].fillna(sub_118_tc[["age", "is_female"]].mean()).values

    # 2. Pooled GSE118553 Cortical (TC + EC + FC)
    sub_118_cortical = meta_118[(meta_118["region"].isin(["Temporal_Cortex", "Entorhinal_Cortex", "Frontal_Cortex"])) & 
                                (meta_118["diagnosis"].isin(["CTL", "AD"]))].copy()
    sub_118_cortical["is_female"] = (sub_118_cortical["sex"] == "FEMALE").astype(float)
    e_118_cortical = expr_118[sub_118_cortical.index].T
    y_118_cortical = (sub_118_cortical["diagnosis"] == "AD").astype(int).values
    donors_118_cortical = sub_118_cortical["donor_id"].values
    cov_118_cortical = sub_118_cortical[["age", "is_female"]].fillna(sub_118_cortical[["age", "is_female"]].mean()).values

    # 3. Lockbox GSE132903 MTG
    sub_132 = meta_132[meta_132["diagnosis"].isin(["CTL", "AD"])].copy()
    sub_132["is_female"] = (sub_132["sex"] == "FEMALE").astype(float)
    e_132 = expr_132[sub_132.index].T
    y_132 = (sub_132["diagnosis"] == "AD").astype(int).values
    donors_132 = sub_132["donor_id"].values
    cov_132 = sub_132[["age", "is_female"]].fillna(sub_132[["age", "is_female"]].mean()).values

    # 4. MSBB Temporal Cortex
    temp_msbb = meta_msbb[meta_msbb["region"].isin(["BM22_STG", "BM20_ITG", "BM38_TP"])].dropna(subset=["braak"]).copy()
    temp_msbb["is_ad"] = (temp_msbb["braak"] >= 4).astype(int)
    e_msbb_temp = expr_msbb[temp_msbb.index]

    benchmark_configs = [
        {
            "id": "baseline_h3",
            "name": "Baseline H3 (TC, 1000 HVG, Z-score)",
            "data_mode": "single_tc",
            "feature_strategy": "hvg",
            "top_k": 1000,
            "transform": "zscore",
        },
        {
            "id": "biological_priors_only",
            "name": "Curated Biological Priors Only (~70 genes)",
            "data_mode": "single_tc",
            "feature_strategy": "priors_only",
            "top_k": None,
            "transform": "zscore",
        },
        {
            "id": "hybrid_priors",
            "name": "Hybrid Priors (TC, 500 HVG + Biological Priors)",
            "data_mode": "single_tc",
            "feature_strategy": "hybrid",
            "top_k": 500,
            "transform": "zscore",
        },
        {
            "id": "rank_transform_only",
            "name": "Rank Transform Only (TC, 1000 HVG, Within-Sample Rank)",
            "data_mode": "single_tc",
            "feature_strategy": "hvg",
            "top_k": 1000,
            "transform": "rank",
        },
        {
            "id": "multi_region_pooled",
            "name": "Multi-Region Cortical Pooled (TC+EC+FC, 1000 HVG)",
            "data_mode": "multi_cortical",
            "feature_strategy": "hvg",
            "top_k": 1000,
            "transform": "zscore",
        },
        {
            "id": "composite_a_pooled_hybrid",
            "name": "Composite Model A (Pooled Cortical + Hybrid Priors)",
            "data_mode": "multi_cortical",
            "feature_strategy": "hybrid",
            "top_k": 500,
            "transform": "zscore",
        },
        {
            "id": "composite_b_pooled_rank",
            "name": "Composite Model B (Pooled Cortical + Rank Transform)",
            "data_mode": "multi_cortical",
            "feature_strategy": "hvg",
            "top_k": 1000,
            "transform": "rank",
        },
        {
            "id": "composite_c_full_synergy",
            "name": "Composite Model C (Pooled + Hybrid Priors + Rank Transform)",
            "data_mode": "multi_cortical",
            "feature_strategy": "hybrid",
            "top_k": 500,
            "transform": "rank",
        },
    ]

    all_results = []
    saved_model_weights = {}
    saved_predictions = {"y_lockbox": y_132.tolist(), "preds": {}}

    for cfg in benchmark_configs:
        print(f"\nEvaluating: {cfg['name']}...")
        if cfg["data_mode"] == "single_tc":
            e_tr_full = e_118_tc
            y_tr_full = y_118_tc
            donors_tr = donors_118_tc
            cov_tr_full = cov_118_tc
        else:
            e_tr_full = e_118_cortical
            y_tr_full = y_118_cortical
            donors_tr = donors_118_cortical
            cov_tr_full = cov_118_cortical

        def fit_and_predict(X_tr_df, y_tr_arr, X_eval_df, transform_mode, feat_strat, top_k):
            # Select features strictly on train
            if feat_strat == "priors_only":
                genes = [g for g in CANONICAL_BIOLOGICAL_PRIORS if g in X_tr_df.columns]
            elif feat_strat == "hybrid":
                priors = [g for g in CANONICAL_BIOLOGICAL_PRIORS if g in X_tr_df.columns]
                hvg = X_tr_df.var(axis=0).nlargest(top_k).index.tolist()
                genes = list(dict.fromkeys(hvg + priors))
            else:
                genes = X_tr_df.var(axis=0).nlargest(top_k).index.tolist()

            X_tr_sub = X_tr_df[genes].copy()
            # Handle out of vocab columns on eval
            X_eval_sub = pd.DataFrame(0.0, index=X_eval_df.index, columns=genes)
            common_cols = [g for g in genes if g in X_eval_df.columns]
            X_eval_sub[common_cols] = X_eval_df[common_cols]

            if transform_mode == "rank":
                X_tr_sub = X_tr_sub.rank(axis=1, pct=True)
                X_eval_sub = X_eval_sub.rank(axis=1, pct=True)

            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr_sub.values)
            X_eval_sc = scaler.transform(X_eval_sub.values)

            clf = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
            clf.fit(X_tr_sc, y_tr_arr)
            preds = clf.predict_proba(X_eval_sc)[:, 1]
            return preds, clf, genes, scaler

        # 1. 5-Fold Donor-Grouped CV
        splits = get_donor_grouped_splits(donors_tr, y=y_tr_full, n_splits=5, seed=42)
        cv_preds = np.full(len(e_tr_full), np.nan)
        for tr_idx, te_idx in splits:
            fold_preds, _, _, _ = fit_and_predict(
                e_tr_full.iloc[tr_idx], y_tr_full[tr_idx], e_tr_full.iloc[te_idx],
                cfg["transform"], cfg["feature_strategy"], cfg["top_k"]
            )
            cv_preds[te_idx] = fold_preds

        cv_auc = calc_auc(y_tr_full, cv_preds)

        # 2. Fit Full Model & Predict on Lockbox
        lb_preds, full_clf, sel_genes, full_sc = fit_and_predict(
            e_tr_full, y_tr_full, e_132,
            cfg["transform"], cfg["feature_strategy"], cfg["top_k"]
        )
        lb_auc = calc_auc(y_132, lb_preds)
        saved_predictions["preds"][cfg["id"]] = lb_preds.tolist()

        # Save signed model coefficients
        coef_series = pd.Series(full_clf.coef_.flatten(), index=sel_genes)
        saved_model_weights[cfg["id"]] = coef_series.to_dict()

        # 3. Donor-Cluster Bootstrap on Lockbox (B=2,000)
        df_lb_eval = pd.DataFrame({
            "y_true": y_132,
            "pred": lb_preds,
            "donor_id": donors_132
        })
        boot_res = donor_bootstrap_ci(lambda d: calc_auc(d["y_true"].values, d["pred"].values), df_lb_eval, donor_col="donor_id", n_boot=2000, seed=42)
        ci_low = boot_res["ci_lower"]
        ci_high = boot_res["ci_upper"]

        # 4. Permutation Test (B=1,000)
        perm_res = donor_permutation_test(lambda d: calc_auc(d["y_true"].values, d["pred"].values), df_lb_eval, label_col="y_true", donor_col="donor_id", n_perm=1000, seed=42)
        perm_p = perm_res["p_value"]

        # 5. Age/Sex In-Fold Residualized Evaluation
        e_tr_resid, e_132_resid = residualize_covariates(e_tr_full.values, cov_tr_full, e_132.values, cov_132)
        df_tr_resid = pd.DataFrame(e_tr_resid, index=e_tr_full.index, columns=e_tr_full.columns)
        df_132_resid = pd.DataFrame(e_132_resid, index=e_132.index, columns=e_132.columns)

        lb_preds_resid, _, _, _ = fit_and_predict(
            df_tr_resid, y_tr_full, df_132_resid,
            cfg["transform"], cfg["feature_strategy"], cfg["top_k"]
        )
        lb_auc_resid = calc_auc(y_132, lb_preds_resid)

        res_dict = {
            "id": cfg["id"],
            "name": cfg["name"],
            "data_mode": cfg["data_mode"],
            "n_train_samples": len(e_tr_full),
            "n_train_donors": len(np.unique(donors_tr)),
            "n_features": len(sel_genes),
            "cv_auc": round(cv_auc, 4),
            "lockbox_auc": round(lb_auc, 4),
            "lockbox_ci_lower": round(ci_low, 4),
            "lockbox_ci_upper": round(ci_high, 4),
            "permutation_p": round(perm_p, 4),
            "residualized_lockbox_auc": round(lb_auc_resid, 4),
        }
        all_results.append(res_dict)
        print(f"  -> Internal CV: {cv_auc:.3f} | Lockbox AUC: {lb_auc:.3f} [{ci_low:.3f}, {ci_high:.3f}] (p={perm_p:.3e}) | Resid AUC: {lb_auc_resid:.3f}")

    # 6. Singscore Cross-Platform Transfer Sweep on MSBB
    print("\nEvaluating Singscore Cross-Platform Transfer on MSBB RNA-seq...")
    ad_idx = np.where(y_118_tc == 1)[0]
    ctl_idx = np.where(y_118_tc == 0)[0]
    t_stats, _ = stats.ttest_ind(e_118_tc.iloc[ad_idx], e_118_tc.iloc[ctl_idx], axis=0, equal_var=False)
    t_series = pd.Series(t_stats, index=e_118_tc.columns).dropna().sort_values(ascending=False)

    singscore_sweep = []
    for k in [10, 25, 50, 100, 200, 300]:
        up = t_series.head(k).index.tolist()
        down = t_series.tail(k).index.tolist()
        scores = score_gene_signature_singscore(e_msbb_temp, up_genes=up, down_genes=down)
        auc_val = calc_auc(temp_msbb["is_ad"].values, scores.values)
        df_msbb_eval = pd.DataFrame({
            "y_true": temp_msbb["is_ad"].values,
            "pred": scores.values,
            "donor_id": temp_msbb["donor_id"].values
        })
        b_res = donor_bootstrap_ci(lambda d: calc_auc(d["y_true"].values, d["pred"].values), df_msbb_eval, donor_col="donor_id", n_boot=1000, seed=42)
        singscore_sweep.append({"k": k, "auc": round(auc_val, 4), "ci_lower": round(b_res["ci_lower"], 4), "ci_upper": round(b_res["ci_upper"], 4)})
        print(f"  Singscore K={k:3d}: MSBB AUC = {auc_val:.3f} [{b_res['ci_lower']:.3f}, {b_res['ci_upper']:.3f}]")

    # Save complete results
    out_payload = {
        "benchmark_summary": all_results,
        "singscore_sweep": singscore_sweep,
        "model_weights": saved_model_weights,
        "predictions": saved_predictions
    }

    os.makedirs("results", exist_ok=True)
    with open("results/model_ablation_results.json", "w") as f:
        json.dump(out_payload, f, indent=2)

    df_summary = pd.DataFrame(all_results)
    df_summary.to_csv("results/table_model_ablation.csv", index=False)

    print("\n" + "=" * 75)
    print("DEEP BENCHMARK COMPLETE: SUMMARY TABLE:")
    print("=" * 75)
    print(df_summary[["name", "n_features", "cv_auc", "lockbox_auc", "lockbox_ci_lower", "lockbox_ci_upper", "residualized_lockbox_auc"]].to_string(index=False))
    print("\nSaved full results to results/model_ablation_results.json and results/table_model_ablation.csv")


if __name__ == "__main__":
    run_deep_benchmark()
