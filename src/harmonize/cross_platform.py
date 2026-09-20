"""
Cross-Platform Feature Harmonization & Rank-Based Scoring.
Implements:
1. Singscore-style within-sample percentile ranking (platform-invariant single-sample scoring)
2. Gene-level standardized z-scoring (within-cohort / in-fold)
3. Quantile normalization and rank transform
"""

import numpy as np
import pandas as pd
from scipy import stats


def compute_sample_ranks(expr_df):
    """
    Transform gene expression into within-sample relative rank percentiles [0, 1].
    Rows: Genes, Columns: Samples.
    Robust to platform shifts, probe dynamic ranges, and batch offsets.
    """
    # Rank each column (sample) along rows (genes)
    ranks = expr_df.rank(axis=0, ascending=True, pct=True)
    return ranks


def score_gene_signature_singscore(expr_df, up_genes, down_genes=None):
    """
    Compute single-sample directional signature score using rank statistics (singscore algorithm).
    up_genes: list of upregulated gene symbols
    down_genes: optional list of downregulated gene symbols
    Returns Series of signature scores indexed by sample.
    """
    ranks = compute_sample_ranks(expr_df)
    n_genes = len(ranks)

    valid_up = [g for g in up_genes if g in ranks.index]
    if not valid_up:
        raise ValueError("No valid up_genes found in expression matrix")

    up_rank_sum = ranks.loc[valid_up].sum(axis=0)
    score_up = (up_rank_sum / len(valid_up)) - 0.5

    if down_genes:
        valid_down = [g for g in down_genes if g in ranks.index]
        if valid_down:
            down_rank_sum = ranks.loc[valid_down].sum(axis=0)
            score_down = (down_rank_sum / len(valid_down)) - 0.5
            return score_up - score_down

    return score_up


def standardize_within_cohort(expr_df):
    """
    Z-score standardizes each gene across samples within a cohort: (x - mean) / std.
    Rows: Genes, Columns: Samples.
    """
    means = expr_df.mean(axis=1)
    stds = expr_df.std(axis=1).replace(0, 1.0)
    z_df = expr_df.sub(means, axis=0).div(stds, axis=0)
    return z_df


def harmonize_features_intersection(expr_dict):
    """
    Intersect genes across multiple expression matrices.
    expr_dict: dict of {cohort_name: expr_df (genes x samples)}
    Returns dict of subsetted expr_dfs with identical row ordering.
    """
    common_genes = None
    for name, df in expr_dict.items():
        if common_genes is None:
            common_genes = set(df.index)
        else:
            common_genes = common_genes.intersection(set(df.index))
    
    common_genes = sorted(list(common_genes))
    return {name: df.loc[common_genes] for name, df in expr_dict.items()}
