"""H2 transfer at the donor level (post hoc, added after external review).
The original H2 transfer AUC scored every MSBB sample (487 samples, 60 donors, up to 19 regions each) and treated
samples as the AUC unit. Singscore varies strongly by region, so sample-level pooling mixes between-region and
between-donor variance. Here each donor gets one score (mean over regions; optionally z-scored within region first)
and the AUC unit is the donor. The signature is built exactly as in h2_resilience.py (top/bottom 100 t-statistics,
AsymAD vs AD, GSE118553 temporal cortex). Permutation p is one-sided (resilient donors score higher)."""
import json
import numpy as np
import pandas as pd
from scipy import stats
from src.load import load_gse84422, load_gse118553
from src.analyses.h2_resilience import define_resilience_cohort
from src.harmonize.cross_platform import score_gene_signature_singscore
from src.eval.metrics import calc_auc
from src.eval.ledger_logger import log_analysis


def _delong_ci(y, sc, alpha=0.05):
    """DeLong variance for a single AUC (Sun & Xu fast implementation)."""
    from scipy import stats
    pos, neg = sc[y == 1], sc[y == 0]
    m, n = len(pos), len(neg)
    allv = np.r_[pos, neg]
    def mid(x):
        r = stats.rankdata(x, method="average"); return r
    tz = mid(allv); tx = mid(pos); ty = mid(neg)
    auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
    v01 = (tz[:m] - tx) / n
    v10 = 1 - (tz[m:] - ty) / m
    var = np.var(v01, ddof=1) / m + np.var(v10, ddof=1) / n
    se = np.sqrt(var); z = stats.norm.ppf(1 - alpha / 2)
    return float(max(0, auc - z * se)), float(min(1, auc + z * se)), float(se)


def _auc_ci_p(y, sc, rng, n_boot=5000, n_perm=10000):
    auc = calc_auc(y, sc)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            bs.append(calc_auc(y[i], sc[i]))
    p = float(np.mean([calc_auc(rng.permutation(y), sc) >= auc for _ in range(n_perm)]))
    return float(auc), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), p


def run_h2_donor_level():
    meta, expr = load_gse84422()
    m118, e118 = load_gse118553()
    coh = define_resilience_cohort(meta)
    sub = m118[(m118.region == "Temporal_Cortex") & m118.diagnosis.isin(["AsymAD", "AD"])]
    E = e118[sub.index].T
    asym = (sub.diagnosis == "AsymAD").values
    t = {}
    for g in E.columns:
        tv, _ = stats.ttest_ind(E.loc[asym, g], E.loc[~asym, g], equal_var=False)
        if not np.isnan(tv):
            t[g] = tv
    s = pd.Series(t).sort_values(ascending=False)
    up = [g for g in s.head(100).index if g in expr.index]
    dn = [g for g in s.tail(100).index if g in expr.index]
    coh["score"] = score_gene_signature_singscore(expr[coh.index], up, dn).values
    coh["score_rc"] = coh.groupby("region")["score"].transform(lambda x: (x - x.mean()) / (x.std(ddof=0) or 1.0))
    d = coh.groupby("donor_id").agg(y=("is_resilient", "first"), score=("score", "mean"), score_rc=("score_rc", "mean"),
                                    n_regions=("score", "size"))
    rng = np.random.default_rng(1)
    y = d.y.values
    a, lo, hi, p = _auc_ci_p(y, d.score.values, rng)
    a2, lo2, hi2, p2 = _auc_ci_p(y, d.score_rc.values, rng)
    # pre-specified analysis: sample-level AUC; donor labels permuted (all samples of a donor share the label)
    lab = coh.groupby("donor_id")["is_resilient"].first()
    samp_y = coh.is_resilient.values; samp_d = coh.donor_id.values; samp_s = coh.score.values
    obs = calc_auc(samp_y, samp_s)
    don_idx = {k_: np.where(samp_d == k_)[0] for k_ in lab.index}
    ge = 0
    for _ in range(10000):
        perm = dict(zip(lab.index, rng.permutation(lab.values)))
        yy = np.empty(len(samp_y))
        for dn_, ix in don_idx.items():
            yy[ix] = perm[dn_]
        ge += calc_auc(yy, samp_s) >= obs
    samp_p = float((ge + 1) / 10001)
    dl_lo, dl_hi, dl_se = _delong_ci(y, d.score.values)
    dl_lo2, dl_hi2, _ = _delong_ci(y, d.score_rc.values)
    # continuous, amyloid-aware analysis over ALL donors with CERAD: residual of CDR given Braak, CERAD, age, sex, PMI
    allv = meta.dropna(subset=["braak", "cdr", "cerad", "age", "sex", "pmi", "donor_id"]).copy()
    allv["sc"] = score_gene_signature_singscore(expr[allv.index], up, dn).values
    allv["sc_rc"] = allv.groupby("region")["sc"].transform(lambda x: (x - x.mean()) / (x.std(ddof=0) or 1.0))
    dd = allv.groupby("donor_id").agg(cdr=("cdr", "first"), braak=("braak", "first"), cerad=("cerad", "first"),
                                      age=("age", "first"), sex=("sex", "first"), pmi=("pmi", "first"), sc=("sc", "mean"), sc_rc=("sc_rc", "mean"))
    import statsmodels.api as sm
    Z = sm.add_constant(np.c_[dd.braak, dd.cerad, dd.age, (dd.sex == "FEMALE").astype(float), dd.pmi])
    resid = dd.cdr.values - Z @ np.linalg.lstsq(Z, dd.cdr.values, rcond=None)[0]
    r_sc = stats.spearmanr(dd.sc_rc, resid); r_raw = stats.spearmanr(dd.sc, resid)
    out = {"continuous_n_donors": int(len(dd)),
           "continuous_rho_region_centred": float(r_sc[0]), "continuous_p_region_centred": float(r_sc[1]),
           "continuous_rho_raw": float(r_raw[0]), "continuous_p_raw": float(r_raw[1]),
           "sample_level_perm_p_one_sided": samp_p, "sample_level_obs_auc": float(obs),
           "delong_ci_lower": dl_lo, "delong_ci_upper": dl_hi, "delong_se": dl_se,
           "rc_delong_ci_lower": dl_lo2, "rc_delong_ci_upper": dl_hi2,
           "n_resilient": int(y.sum()), "n_demented": int((1 - y).sum()),
           "sample_level_auc_for_reference": float(calc_auc(coh.is_resilient.values, coh.score.values)),
           "n_samples_sample_level": int(len(coh)),
           "donor_auc": a, "ci_lower": lo, "ci_upper": hi, "perm_p_one_sided": p,
           "donor_auc_region_centred": a2, "rc_ci_lower": lo2, "rc_ci_upper": hi2, "rc_perm_p_one_sided": p2,
           "median_regions_per_donor": float(d.n_regions.median())}
    json.dump(out, open("results/h2_donor_level.json", "w"), indent=2)
    log_analysis(analysis_id="H2_DONOR_001", category="EXPLORATORY_POSTHOC",
                 hypothesis_or_idea="H2 AsymAD-signature transfer at donor level (post hoc, review)",
                 description="One score per donor (mean over regions), donor bootstrap CI and permutation p",
                 cohorts_used="GSE118553 -> GSE84422", regions_analyzed="all MSBB regions with a sample",
                 model_type="Singscore_donor_mean", covariates_adjusted="None", sample_size_donors=int(len(d)),
                 primary_metric="Donor_AUC", point_estimate=a, ci_95_lower=lo, ci_95_upper=hi, p_value=p,
                 replicated_in_lockbox=False, status="COMPLETE",
                 notes="Differs from sample-level 0.612; post hoc, N=7 resilient donors, exploratory.")
    return out


if __name__ == "__main__":
    print(run_h2_donor_level())
