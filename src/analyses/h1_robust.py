"""H1 robustness (post hoc, added after external review).
Addresses: (i) each regional score came from ONE 5-fold split; (ii) the permutation test treated the 19 regional
estimates as fixed although their sampling error is ~76% of their between-region variance; (iii) no age/PMI/sex
adjustment in H1.
Design: for every region, R repeated donor-grouped 5-fold CV runs (random donor->fold assignment per repeat) of the
same ridge model as H1; predictions are averaged over repeats. Two variants: 'plain' (as H1) and 'adjusted'
(age, sex, PMI regressed out in-fold, coefficients fitted on the training fold only). Regional score = Spearman rho
between averaged out-of-fold prediction and Braak. Then:
  * tier correlation rho_H1 and its one-sided tier-label permutation p (5,000 shuffles) for each variant;
  * donor bootstrap (2,000): donors are resampled GLOBALLY (same resample for every region, preserving donor sharing
    across regions), each region's rho is recomputed from its stored out-of-fold predictions, and the tier
    correlation is recomputed. This propagates regional sampling error into a CI for rho_H1 (it does not re-fit models).
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from src.load import load_gse84422
from src.analyses.h1_regional import PRE_REGISTERED_RANKS, _rank_corr_perm_p
from src.models.linear_pipeline import InFoldLinearModel, residualize_covariates
from src.eval.ledger_logger import log_analysis

R = 10


def _folds(donors, seed, k=5):
    u = np.array(sorted(set(donors)))
    rng = np.random.default_rng(seed)
    rng.shuffle(u)
    fold_of = {d: i % k for i, d in enumerate(u)}
    f = np.array([fold_of[d] for d in donors])
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]


def run_h1_robust():
    meta, expr = load_gse84422()
    meta = meta.dropna(subset=["braak", "donor_id"])
    regions = [r for r, n in meta.region.value_counts().items() if n >= 15]
    store = {}
    for reg in regions:
        sub = meta[meta.region == reg]
        X = expr[sub.index].T
        y = sub.braak.values.astype(float)
        d = sub.donor_id.values
        cov = pd.DataFrame({"age": sub.age, "pmi": sub.pmi, "sex": (sub.sex == "FEMALE").astype(float)})
        cov = cov.fillna(cov.median()).values
        preds = {"plain": np.zeros((R, len(sub))), "adjusted": np.zeros((R, len(sub)))}
        for r in range(R):
            for tr, te in _folds(d, seed=100 + r):
                m = InFoldLinearModel("ridge_regression", top_hvg=1000, seed=42).fit_transform_train(X.iloc[tr], y[tr])
                preds["plain"][r, te] = m.predict(X.iloc[te])
                Xtr, Xte = residualize_covariates(X.iloc[tr].values, cov[tr], X.iloc[te].values, cov[te])
                m2 = InFoldLinearModel("ridge_regression", top_hvg=1000, seed=42).fit_transform_train(
                    pd.DataFrame(Xtr, columns=X.columns), y[tr])
                preds["adjusted"][r, te] = m2.predict(pd.DataFrame(Xte, columns=X.columns))
        store[reg] = dict(y=y, d=d, p={k: v.mean(0) for k, v in preds.items()},
                          rho_each={k: [stats.spearmanr(y, v[i])[0] for i in range(R)] for k, v in preds.items()})
        print(reg, "done", flush=True)

    tiers = np.array([PRE_REGISTERED_RANKS.get(r, 3) for r in regions]); vuln = 6 - tiers
    all_donors = sorted(set(meta.donor_id))
    rng = np.random.default_rng(7)
    out = {"n_repeats": R, "n_regions": len(regions), "variants": {}}
    for var in ["plain", "adjusted"]:
        rho_reg = np.array([stats.spearmanr(store[r]["y"], store[r]["p"][var])[0] for r in regions])
        rho_h1, p_perm = _rank_corr_perm_p(vuln, rho_reg, n_perm=5000, seed=42)
        # spread of the regional score across single repeats (how much one split moves each region)
        within_sd = float(np.mean([np.std(store[r]["rho_each"][var]) for r in regions]))
        boots = []
        for _ in range(2000):
            samp = rng.choice(all_donors, len(all_donors), replace=True)
            cnt = pd.Series(samp).value_counts().to_dict()
            rr = []
            for r in regions:
                w = np.array([cnt.get(dn, 0) for dn in store[r]["d"]])
                idx = np.repeat(np.arange(len(w)), w)
                rr.append(stats.spearmanr(store[r]["y"][idx], store[r]["p"][var][idx])[0] if len(set(store[r]["y"][idx])) > 1 else np.nan)
            rr = np.array(rr)
            if np.isnan(rr).any():
                continue
            boots.append(stats.spearmanr(vuln, rr)[0])
        boots = np.array(boots)
        # per-region 95% donor-bootstrap CI of the repeated-CV score
        reg_ci = {}
        for r in regions:
            yv, pv = store[r]["y"], store[r]["p"][var]
            bb = []
            for _ in range(1000):
                ix = rng.integers(0, len(yv), len(yv))
                if len(set(yv[ix])) > 1:
                    bb.append(stats.spearmanr(yv[ix], pv[ix])[0])
            reg_ci[r] = [float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))]
        # array-type sensitivity: drop regions measured on HG-U133 Plus 2 (platform is constant within a region)
        plat = meta.groupby("region")["platform"].agg(lambda x: x.mode()[0])
        keep = [i for i, r in enumerate(regions) if plat[r] != "GPL570"]
        rho_k, p_k = _rank_corr_perm_p(vuln[keep], rho_reg[keep], n_perm=5000, seed=42)
        out["variants"][var] = {
            "regional_ci": reg_ci,
            "platform_excluded_regions": [r for r in regions if plat[r] == "GPL570"],
            "rho_H1_excl_plus2": float(rho_k), "perm_p_excl_plus2": float(p_k), "n_regions_excl_plus2": int(len(keep)),
            "regional_rho": dict(zip(regions, map(float, rho_reg))),
            "rho_H1": float(rho_h1), "perm_p": float(p_perm),
            "boot_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "boot_frac_rho_gt_0": float((boots > 0).mean()), "n_boot_valid": int(len(boots)),
            "mean_within_region_sd_across_repeats": within_sd,
        }
        print(var, out["variants"][var]["rho_H1"], out["variants"][var]["perm_p"], out["variants"][var]["boot_ci"], flush=True)
    json.dump(out, open("results/h1_robust.json", "w"), indent=2)
    v = out["variants"]["adjusted"]
    log_analysis(analysis_id="H1_ROBUST_001", category="SENSITIVITY",
                 hypothesis_or_idea="H1 repeated-CV, age/sex/PMI-adjusted, donor-bootstrapped (post hoc, review)",
                 description="10 repeated donor-grouped CVs per region; in-fold covariate residualisation; global donor bootstrap through the tier correlation",
                 cohorts_used="GSE84422_MSBB", regions_analyzed=f"{len(regions)} regions", model_type="InFold_Ridge_repeatedCV",
                 covariates_adjusted="Age, Sex, PMI (adjusted variant)", sample_size_donors=len(all_donors),
                 primary_metric="Spearman_rho_tier", point_estimate=v["rho_H1"], ci_95_lower=v["boot_ci"][0],
                 ci_95_upper=v["boot_ci"][1], p_value=v["perm_p"], replicated_in_lockbox=False, status="COMPLETE",
                 notes=f"plain rho={out['variants']['plain']['rho_H1']:.3f}")
    return out


if __name__ == "__main__":
    run_h1_robust()
