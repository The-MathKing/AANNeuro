"""
Sparse Biomarker Panel Selection using Elastic Net + Stability Selection.
Subsamples donors across 100 iterations to identify the most robust <= 20 gene signatures.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.preprocessing import StandardScaler


def stability_selection_panel(X_df, y_series, donor_series, n_subsamples=100, max_genes=20, l1_ratio=0.7, C=0.5, seed=42):
    """
    Perform stability selection across donor-grouped subsampling.
    X_df: DataFrame (samples x genes)
    y_series: Series of binary labels (0 / 1)
    donor_series: Series of donor IDs
    Returns: DataFrame of selected genes ranked by selection probability.
    """
    rng = np.random.default_rng(seed)
    unique_donors = donor_series.unique()
    n_sub_donors = int(len(unique_donors) * 0.75)
    
    genes = X_df.columns.values
    n_genes = len(genes)
    selection_counts = np.zeros(n_genes, dtype=int)

    donor_to_indices = X_df.groupby(donor_series).indices

    for i in range(n_subsamples):
        sampled_donors = rng.choice(unique_donors, size=n_sub_donors, replace=False)
        sampled_idx = np.concatenate([donor_to_indices[d] for d in sampled_donors])
        
        X_sub = X_df.iloc[sampled_idx].values
        y_sub = y_series.iloc[sampled_idx].values
        
        if len(np.unique(y_sub)) < 2:
            continue
            
        scaler = StandardScaler()
        X_sub_scaled = scaler.fit_transform(X_sub)
        
        # Fit L1/ElasticNet logistic regression
        clf = LogisticRegression(
            penalty='elasticnet', l1_ratio=l1_ratio, C=C, solver='saga', max_iter=500, random_state=seed + i
        )
        try:
            clf.fit(X_sub_scaled, y_sub)
            coefs = np.abs(clf.coef_.flatten())
            selected = coefs > 1e-4
            selection_counts += selected.astype(int)
        except Exception:
            continue

    probabilities = selection_counts / n_subsamples
    res_df = pd.DataFrame({
        "gene": genes,
        "selection_probability": probabilities
    }).sort_values("selection_probability", ascending=False)

    top_panel = res_df.head(max_genes)
    return top_panel
