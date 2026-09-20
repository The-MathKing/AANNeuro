"""
Donor-Grouped Cross-Validation.
Ensures ZERO donor leakage across training, validation, and test splits.
All donors in a test fold are completely held out from training and feature transformations.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


def get_donor_grouped_splits(donor_ids, y=None, n_splits=5, seed=42):
    """
    Generate train/test split indices ensuring zero donor overlap.
    donor_ids: array-like of donor identifiers.
    y: optional binary/multiclass target for stratified group splitting.
    """
    donor_ids = np.asarray(donor_ids)
    unique_donors = np.unique(donor_ids)
    
    if len(unique_donors) < n_splits:
        n_splits = max(2, len(unique_donors))

    if y is not None and len(np.unique(y[~pd.isna(y)])) > 1:
        # StratifiedGroupKFold
        try:
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            splits = list(sgkf.split(X=np.zeros(len(donor_ids)), y=y, groups=donor_ids))
        except Exception:
            gkf = GroupKFold(n_splits=n_splits)
            splits = list(gkf.split(X=np.zeros(len(donor_ids)), groups=donor_ids))
    else:
        gkf = GroupKFold(n_splits=n_splits)
        splits = list(gkf.split(X=np.zeros(len(donor_ids)), groups=donor_ids))

    # Verify zero donor overlap in every split
    for fold, (train_idx, test_idx) in enumerate(splits):
        train_donors = set(donor_ids[train_idx])
        test_donors = set(donor_ids[test_idx])
        overlap = train_donors.intersection(test_donors)
        if len(overlap) > 0:
            raise ValueError(f"FATAL LEAKAGE: Fold {fold} has {len(overlap)} overlapping donors: {overlap}")

    return splits
