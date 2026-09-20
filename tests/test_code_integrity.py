"""
Integrity Tests: Code Purity and Data Loading Honesty.
Checks that no synthetic, simulated, or pseudo-random data generators exist in src/load/.
"""

import os
import glob
import pytest


def test_no_synthetic_generators_in_loaders():
    loader_files = glob.glob("src/load/*.py")
    assert len(loader_files) > 0, "No loader files found in src/load/"

    forbidden_patterns = [
        "np.random",
        "random.random",
        "torch.randn",
        "torch.rand",
        "make_classification",
        "make_blobs",
        "simulat"
    ]

    for fpath in loader_files:
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read().lower()
            lines = content.splitlines()

        for line_no, line in enumerate(lines, 1):
            # Allow comments if explicitly clarifying no simulation
            if line.strip().startswith("#") or line.strip().startswith('"""') or line.strip().startswith("'''"):
                continue
            for pattern in forbidden_patterns:
                assert pattern not in line, (
                    f"Forbidden generator '{pattern}' found in {fpath}:{line_no} -> '{line}'"
                )
