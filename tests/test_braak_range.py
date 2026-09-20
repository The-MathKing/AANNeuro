"""Guard against out-of-range Braak codes (Draft 9 audit: donor 1009 had Braak = 9 in GSE84422)."""

import numpy as np
from src.load import load_gse84422
from src.analyses.h2_resilience import define_resilience_cohort


def test_braak_within_defined_scale():
    meta, _ = load_gse84422()
    b = meta["braak"].dropna()
    assert b.between(0, 6).all(), f"Braak values outside 0-6: {sorted(set(b[~b.between(0, 6)]))}"


def test_high_pathology_group_stats_are_feasible():
    meta, _ = load_gse84422()
    coh = define_resilience_cohort(meta)
    per_donor = coh.groupby("donor_id")["braak"].first()
    assert per_donor.between(4, 6).all()
    for _, grp in coh.groupby("is_resilient"):
        g = grp.groupby("donor_id")["braak"].first()
        n = len(g)
        # the largest SD attainable for n values confined to [4, 6] (half at each end)
        max_sd = np.sqrt((2.0 ** 2) * n / (4 * (n - 1))) if n > 1 else 0.0
        assert g.std() <= max_sd + 1e-9
