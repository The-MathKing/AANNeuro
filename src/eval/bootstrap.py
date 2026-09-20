"""
Donor-Level Cluster Bootstrap for Rigorous Uncertainty Quantification.
Resamples entire donors with replacement, preserving within-donor multi-region correlation structures.
Calculates 95% Confidence Intervals (2.5th and 97.5th percentiles) across B=2000 resamples.
"""

import numpy as np
import pandas as pd
from src.eval.metrics import calc_auc, calc_spearman, calc_pearson


def donor_bootstrap_ci(eval_fn, df_samples, donor_col="donor_id", n_boot=2000, seed=42, ci=0.95):
    """
    Compute 95% bootstrap confidence interval by resampling donors.
    eval_fn: function taking sampled df and returning float or dict of floats.
    df_samples: DataFrame containing sample-level predictions and donor_col.
    """
    rng = np.random.default_rng(seed)
    unique_donors = df_samples[donor_col].unique()
    n_donors = len(unique_donors)

    # Pre-group indices by donor for rapid sampling
    donor_to_indices = df_samples.groupby(donor_col).indices

    # Initial evaluation on observed data
    observed = eval_fn(df_samples)
    is_dict = isinstance(observed, dict)

    boot_results = []

    for _ in range(n_boot):
        sampled_donors = rng.choice(unique_donors, size=n_donors, replace=True)
        sampled_idx_list = [donor_to_indices[d] for d in sampled_donors]
        sampled_idx = np.concatenate(sampled_idx_list)
        sampled_df = df_samples.iloc[sampled_idx]
        
        try:
            val = eval_fn(sampled_df)
            boot_results.append(val)
        except Exception:
            continue

    alpha_low = (1.0 - ci) / 2.0 * 100
    alpha_high = (1.0 + ci) / 2.0 * 100

    if is_dict:
        out = {}
        for k in observed.keys():
            k_vals = [r[k] for r in boot_results if k in r and not np.isnan(r[k])]
            if len(k_vals) > 0:
                out[k] = {
                    "estimate": float(observed[k]),
                    "ci_lower": float(np.percentile(k_vals, alpha_low)),
                    "ci_upper": float(np.percentile(k_vals, alpha_high)),
                    "boot_std": float(np.std(k_vals))
                }
            else:
                out[k] = {"estimate": float(observed[k]), "ci_lower": np.nan, "ci_upper": np.nan, "boot_std": np.nan}
        return out
    else:
        valid_vals = [v for v in boot_results if not np.isnan(v)]
        if len(valid_vals) > 0:
            return {
                "estimate": float(observed),
                "ci_lower": float(np.percentile(valid_vals, alpha_low)),
                "ci_upper": float(np.percentile(valid_vals, alpha_high)),
                "boot_std": float(np.std(valid_vals))
            }
        return {"estimate": float(observed), "ci_lower": np.nan, "ci_upper": np.nan, "boot_std": np.nan}


def donor_bootstrap_auc_ci(df_preds, y_true_col="y_true", y_pred_col="y_pred", donor_col="donor_id", n_boot=2000, seed=42):
    """Convenience function for donor-bootstrapped AUC CI."""
    def _eval(df):
        return calc_auc(df[y_true_col].values, df[y_pred_col].values)
    return donor_bootstrap_ci(_eval, df_preds, donor_col=donor_col, n_boot=n_boot, seed=seed)


def donor_bootstrap_spearman_ci(df_preds, y_true_col="y_true", y_pred_col="y_pred", donor_col="donor_id", n_boot=2000, seed=42):
    """Convenience function for donor-bootstrapped Spearman rho CI."""
    def _eval(df):
        rho, _ = calc_spearman(df[y_true_col].values, df[y_pred_col].values)
        return rho
    return donor_bootstrap_ci(_eval, df_preds, donor_col=donor_col, n_boot=n_boot, seed=seed)
