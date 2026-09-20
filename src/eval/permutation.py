"""
Donor-Level Permutation Testing.
Permutes labels across unique donors to preserve within-donor multi-tissue dependencies.
Computes exact empirical p-values as (sum(T_perm >= T_obs) + 1) / (N_perm + 1).
"""

import numpy as np
import pandas as pd
from src.eval.metrics import calc_auc, calc_spearman


def donor_permutation_test(eval_fn, df_samples, label_col="y_true", donor_col="donor_id", n_perm=5000, seed=42):
    """
    Perform donor-level permutation test.
    eval_fn: function taking df_samples with permuted label_col and returning scalar test statistic.
    df_samples: DataFrame containing data.
    """
    rng = np.random.default_rng(seed)
    
    # Extract unique donor labels
    donor_labels = df_samples.groupby(donor_col)[label_col].first()
    unique_donors = donor_labels.index.values
    orig_labels = donor_labels.values
    
    t_obs = eval_fn(df_samples)
    
    perm_stats = []
    df_copy = df_samples.copy()

    for _ in range(n_perm):
        shuffled_labels = rng.permutation(orig_labels)
        donor_to_perm_label = dict(zip(unique_donors, shuffled_labels))
        df_copy[label_col] = df_copy[donor_col].map(donor_to_perm_label)
        
        try:
            t_p = eval_fn(df_copy)
            perm_stats.append(t_p)
        except Exception:
            continue

    perm_stats = np.array(perm_stats)
    n_valid = len(perm_stats)
    
    if n_valid == 0:
        return {"observed": float(t_obs), "p_value": 1.0, "n_perm": 0}

    # Two-sided or one-sided depending on observed sign
    if t_obs >= 0:
        count_extreme = np.sum(perm_stats >= t_obs)
    else:
        count_extreme = np.sum(perm_stats <= t_obs)

    p_val = float((count_extreme + 1) / (n_valid + 1))
    
    return {
        "observed": float(t_obs),
        "p_value": p_val,
        "n_perm": n_valid,
        "perm_null_mean": float(np.mean(perm_stats)),
        "perm_null_std": float(np.std(perm_stats))
    }
