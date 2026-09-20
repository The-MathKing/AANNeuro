"""
Integrity Tests: Raw Data Checksums Verification.
"""

import os
import hashlib
import pytest


def test_raw_data_checksums():
    checksum_file = "data/raw/checksums.sha256"
    assert os.path.exists(checksum_file), f"Checksum file missing at {checksum_file}"

    with open(checksum_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        expected_hash, fname = line.split(maxsplit=1)
        fpath = os.path.join("data/raw", fname)
        assert os.path.exists(fpath), f"Raw file {fname} not found at {fpath}"

        h = hashlib.sha256()
        with open(fpath, "rb") as f_in:
            while chunk := f_in.read(8192 * 1024):
                h.update(chunk)
        actual_hash = h.hexdigest()

        assert actual_hash == expected_hash, (
            f"Checksum mismatch for {fname}!\nExpected: {expected_hash}\nActual:   {actual_hash}"
        )
