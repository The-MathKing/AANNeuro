"""
Exploratory Analysis E7: Cross-Region Transfer Matrix.
Trains a predictive disease classifier on Region A and evaluates performance on Region B in strictly held-out donors.
Generates an N x N generalization matrix to quantify shared vs region-specific transcriptional programs.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.load import load_gse84422
from src.eval.metrics import calc_auc
from src.eval.ledger_logger import log_analysis
from src.models.linear_pipeline import InFoldLinearModel


N_SPLITS = 20


def run_e7_analysis():
    print("=== RUNNING EXPLORATORY ANALYSIS E7: CROSS-REGION TRANSFER MATRIX ===")
    os.makedirs("results", exist_ok=True)

    meta_msbb, expr_msbb = load_gse84422()
    # Representative regions across vulnerability tiers. Names must match the loader's region labels exactly
    # (an earlier version used BM17_PVC / BM4_PMC, which do not exist in the loader output, so those rows/columns were all-NaN).
    target_regions = ["BM36_PHG", "BM22_STG", "BM20_ITG", "BM9_DLPFC", "BM10_FP", "BM7_SPL", "Occipital_Visual_Cortex", "Precentral_Gyrus"]
    missing = [r for r in target_regions if r not in set(meta_msbb["region"])]
    assert not missing, f"E7 regions absent from data: {missing}"

    valid = meta_msbb[meta_msbb["region"].isin(target_regions)].dropna(subset=["braak", "donor_id"]).copy()
    valid["is_ad"] = (valid["braak"] >= 4).astype(int)
    unique_donors = valid["donor_id"].unique()

    n = len(target_regions)
    mats = np.full((N_SPLITS, n, n), np.nan)
    for k in range(N_SPLITS):
        # 50/50 donor split, re-drawn each repeat; zero donor overlap between the train and test halves within a repeat
        rng = np.random.default_rng(1000 + k)
        shuffled = rng.permutation(unique_donors)
        train_donors = set(shuffled[: len(shuffled) // 2]); test_donors = set(shuffled[len(shuffled) // 2:])
        assert not (train_donors & test_donors)
        for i, r_train in enumerate(target_regions):
            sub_tr = valid[(valid["region"] == r_train) & (valid["donor_id"].isin(train_donors))]
            if len(sub_tr) < 10 or sub_tr["is_ad"].nunique() < 2:
                continue
            clf = InFoldLinearModel(model_type="ridge_classifier", top_hvg=500, seed=42)
            clf.fit_transform_train(expr_msbb[sub_tr.index].T, sub_tr["is_ad"].values)
            for j, r_test in enumerate(target_regions):
                sub_te = valid[(valid["region"] == r_test) & (valid["donor_id"].isin(test_donors))]
                if len(sub_te) < 5 or sub_te["is_ad"].nunique() < 2:
                    continue
                mats[k, i, j] = calc_auc(sub_te["is_ad"].values, clf.predict(expr_msbb[sub_te.index].T))

    mean_matrix = pd.DataFrame(np.nanmean(mats, axis=0), index=target_regions, columns=target_regions)
    offdiag = ~np.eye(n, dtype=bool)
    within = np.array([np.nanmean(np.diag(m)) for m in mats])
    cross = np.array([np.nanmean(m[offdiag]) for m in mats])
    diff = within - cross
    print("\nCross-Region Transfer AUC (mean over %d donor splits; rows = train region, cols = test region):" % N_SPLITS)
    print(mean_matrix.round(3).to_string())
    print(f"\nWithin-region AUC {within.mean():.3f} (sd over splits {within.std():.3f}); cross-region {cross.mean():.3f} (sd {cross.std():.3f}); "
          f"paired within-cross diff {diff.mean():+.3f} (2.5-97.5 pct over splits [{np.percentile(diff, 2.5):+.3f}, {np.percentile(diff, 97.5):+.3f}])")

    results = {
        "analysis": "E7_Cross_Region_Transfer_Matrix",
        "regions": target_regions,
        "n_splits": N_SPLITS,
        "n_donors": int(len(unique_donors)),
        "transfer_matrix": mean_matrix.to_dict(),
        "mean_within_region_auc": float(within.mean()),
        "mean_cross_region_auc": float(cross.mean()),
        "within_auc_sd_over_splits": float(within.std()),
        "cross_auc_sd_over_splits": float(cross.std()),
        "diff_within_minus_cross_mean": float(diff.mean()),
        "diff_pct2_5": float(np.percentile(diff, 2.5)), "diff_pct97_5": float(np.percentile(diff, 97.5)),
        "n_nan_cells": int(np.isnan(mean_matrix.values).sum()),
        "interpretation": "No difference detected between within- and cross-region AUC; both are weak (near 0.55-0.6). Repeated splits share donors, so the interval is descriptive, not a formal test."
    }
    with open("results/e7_transfer_matrix.json", "w") as f:
        json.dump(results, f, indent=2)

    log_analysis(
        analysis_id="E7_XREG_002",
        category="EXPLORATORY",
        hypothesis_or_idea="E7 Cross-Region Transfer Matrix (region-name bug fixed; 20 repeated donor splits)",
        description="Pairwise cross-region transfer of Braak>=4 classifiers in held-out donors; mean over repeated 50/50 donor splits",
        cohorts_used="GSE84422_MSBB",
        regions_analyzed="8 representative regions",
        model_type="Ridge_Cross_Region_Transfer",
        covariates_adjusted="Held-out donor partition",
        sample_size_donors=int(len(unique_donors)),
        primary_metric="Within_minus_Cross_AUC",
        point_estimate=float(diff.mean()),
        ci_95_lower=float(np.percentile(diff, 2.5)),
        ci_95_upper=float(np.percentile(diff, 97.5)),
        p_value=float("nan"),
        replicated_in_lockbox=False,
        status="COMPLETE",
        notes=f"Within {within.mean():.3f} vs cross {cross.mean():.3f}. Interval is over repeated splits (descriptive)."
    )
    return results


if __name__ == "__main__":
    run_e7_analysis()
