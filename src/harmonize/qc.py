"""
Expression Quality Control and Filtering Utilities.
"""

import numpy as np
import pandas as pd


def filter_low_variance_genes(expr_df, top_n=5000):
    """
    Select top N most variable genes across samples based on standard deviation.
    """
    if len(expr_df) <= top_n:
        return expr_df
    variances = expr_df.var(axis=1)
    top_genes = variances.nlargest(top_n).index
    return expr_df.loc[top_genes]


def remove_missing_values(expr_df, max_missing_pct=0.2):
    """
    Filter out genes with excessive missing values and impute remaining with gene mean.
    """
    missing_pct = expr_df.isna().mean(axis=1)
    valid_genes = missing_pct[missing_pct <= max_missing_pct].index
    filtered = expr_df.loc[valid_genes]
    # Impute row-wise mean
    imputed = filtered.apply(lambda row: row.fillna(row.mean()), axis=1)
    return imputed
