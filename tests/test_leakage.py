"""
Integrity Tests: Donor Leakage Verification.
Ensures zero donor overlap between train and test splits across all splits.
"""

import numpy as np
import pytest
from src.load import load_gse118553, load_gse84422
from src.eval.cross_val import get_donor_grouped_splits


def test_zero_donor_leakage_gse118553():
    meta, _ = load_gse118553()
    donor_ids = meta["donor_id"].values
    splits = get_donor_grouped_splits(donor_ids, n_splits=5, seed=42)

    assert len(splits) == 5
    for fold, (train_idx, test_idx) in enumerate(splits):
        train_donors = set(donor_ids[train_idx])
        test_donors = set(donor_ids[test_idx])
        overlap = train_donors.intersection(test_donors)
        assert len(overlap) == 0, f"Leakage detected in fold {fold}: {overlap}"


def test_zero_donor_leakage_gse84422():
    meta, _ = load_gse84422()
    donor_ids = meta["donor_id"].values
    splits = get_donor_grouped_splits(donor_ids, n_splits=5, seed=42)

    assert len(splits) == 5
    for fold, (train_idx, test_idx) in enumerate(splits):
        train_donors = set(donor_ids[train_idx])
        test_donors = set(donor_ids[test_idx])
        overlap = train_donors.intersection(test_donors)
        assert len(overlap) == 0, f"Leakage detected in MSBB fold {fold}: {overlap}"
