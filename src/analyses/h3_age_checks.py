"""
H3 robustness (post-review, Draft 9): is the lockbox AUC an age/sex effect?

Everything here is evaluated on the SAME frozen H3 model (GSE118553 Temporal Cortex, AD vs CTL, top-1000 variance
genes, z-scored, L2 logistic regression with inner-CV C). GSE132903 is used only for evaluation; no lockbox
statistic feeds back into feature selection, scaling or tuning of the H3 model.

Reports:
  1. Age and sex by diagnosis in the lockbox and in the training set.
  2. Age+sex-only classifiers on the lockbox (transferred from GSE118553, and within-lockbox 5-fold CV), plus age alone.
  3. Age/sex-residualised model: residualisation betas fitted on GSE118553 training donors only, applied to the lockbox.
  4. Score-level adjustment inside the lockbox: logistic  AD ~ score + age + sex  (LR test for the score) and its AUC.
  5. Age-matched lockbox subset (1:1 nearest-neighbour on age, same sex, without replacement).
  6. Label-permutation p-value for the lockbox AUC (5,000 permutations).
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.load import load_gse118553, load_gse132903
from src.eval.metrics import calc_auc
from src.eval.bootstrap import donor_bootstrap_ci
from src.eval.ledger_logger import log_analysis
from src.models.linear_pipeline import InFoldLinearModel, residualize_covariates

N_BOOT = 2000
N_PERM = 5000


def _describe(df, label):
    ad, ct = df[df["diagnosis"] == "AD"], df[df["diagnosis"] == "CTL"]
    t, p_age = stats.ttest_ind(ad["age"].dropna(), ct["age"].dropna(), equal_var=False)
    tab = pd.crosstab(df["diagnosis"], df["sex"])
    chi2, p_sex, _, _ = stats.chi2_contingency(tab)
    out = {
        "cohort": label,
        "n_AD": int(len(ad)), "n_CTL": int(len(ct)),
        "age_AD_mean": float(ad["age"].mean()), "age_AD_sd": float(ad["age"].std()),
        "age_CTL_mean": float(ct["age"].mean()), "age_CTL_sd": float(ct["age"].std()),
        "age_AD_min": float(ad["age"].min()), "age_AD_max": float(ad["age"].max()),
        "age_CTL_min": float(ct["age"].min()), "age_CTL_max": float(ct["age"].max()),
        "age_diff_welch_p": float(p_age),
        "female_AD": int((ad["sex"] == "FEMALE").sum()), "female_CTL": int((ct["sex"] == "FEMALE").sum()),
        "sex_chi2_p": float(p_sex),
        "age_missing": int(df["age"].isna().sum()),
    }
    return out


def _boot_auc(y, s, n_boot=N_BOOT, seed=42):
    d = pd.DataFrame({"y_true": y, "y_pred": s, "donor_id": np.arange(len(y))})
    r = donor_bootstrap_ci(lambda x: calc_auc(x["y_true"].values, x["y_pred"].values), d, n_boot=n_boot, seed=seed)
    return float(r["estimate"]), float(r["ci_lower"]), float(r["ci_upper"])


def _perm_p_auc(y, s, n_perm=N_PERM, seed=42):
    rng = np.random.default_rng(seed)
    obs = calc_auc(y, s)
    ge = 0
    for _ in range(n_perm):
        if calc_auc(rng.permutation(y), s) >= obs:
            ge += 1
    return float((ge + 1) / (n_perm + 1)), int(ge)


def _match_age(df, caliper=None):
    """1:1 nearest-neighbour age matching of AD to CTL, exact on sex, without replacement (greedy, closest pairs first)."""
    pairs = []
    for sex in ["FEMALE", "MALE"]:
        ad = df[(df["diagnosis"] == "AD") & (df["sex"] == sex)]
        ct = df[(df["diagnosis"] == "CTL") & (df["sex"] == sex)]
        cand = sorted(((abs(a - c), ia, ic) for ia, a in ad["age"].items() for ic, c in ct["age"].items()))
        used_a, used_c = set(), set()
        for d, ia, ic in cand:
            if ia in used_a or ic in used_c:
                continue
            if caliper is not None and d > caliper:
                continue
            pairs.append((ia, ic, d)); used_a.add(ia); used_c.add(ic)
    return pairs


def run_h3_age_checks():
    print("=== H3 AGE / SEX ROBUSTNESS ON THE LOCKBOX ===")
    os.makedirs("results", exist_ok=True)

    m118, e118 = load_gse118553()
    tr = m118[(m118["region"] == "Temporal_Cortex") & (m118["diagnosis"].isin(["CTL", "AD"]))].dropna(subset=["age"]).copy()
    tr["is_female"] = (tr["sex"] == "FEMALE").astype(float)
    Xtr = e118[tr.index].T
    ytr = (tr["diagnosis"] == "AD").astype(int).values

    m132, e132 = load_gse132903()
    lb = m132[m132["diagnosis"].isin(["CTL", "AD"])].copy()
    lb["is_female"] = (lb["sex"] == "FEMALE").astype(float)
    Xlb = e132[lb.index].T
    ylb = (lb["diagnosis"] == "AD").astype(int).values

    demo_lb = _describe(lb, "GSE132903 lockbox (MTG)")
    tc_all = m118[(m118["region"] == "Temporal_Cortex") & (m118["diagnosis"].isin(["CTL", "AD"]))]
    demo_tr = _describe(tc_all, "GSE118553 Temporal Cortex (training)")
    demo_tr["n_donors"] = int(tc_all["donor_id"].nunique())
    print(f"Lockbox age: AD {demo_lb['age_AD_mean']:.1f}±{demo_lb['age_AD_sd']:.1f} vs CTL {demo_lb['age_CTL_mean']:.1f}±{demo_lb['age_CTL_sd']:.1f} (p={demo_lb['age_diff_welch_p']:.3g}); "
          f"female AD {demo_lb['female_AD']}/{demo_lb['n_AD']} vs CTL {demo_lb['female_CTL']}/{demo_lb['n_CTL']} (p={demo_lb['sex_chi2_p']:.3g})")
    print(f"Training age: AD {demo_tr['age_AD_mean']:.1f}±{demo_tr['age_AD_sd']:.1f} vs CTL {demo_tr['age_CTL_mean']:.1f}±{demo_tr['age_CTL_sd']:.1f} (p={demo_tr['age_diff_welch_p']:.3g})")

    # ---- frozen H3 model (re-fit exactly as in h3_replication.py: all TC CTL/AD samples, incl. those with missing age) ----
    sub = m118[(m118["region"] == "Temporal_Cortex") & (m118["diagnosis"].isin(["CTL", "AD"]))]
    model = InFoldLinearModel(model_type="ridge_classifier", top_hvg=1000, seed=42)
    model.fit_transform_train(e118[sub.index].T, (sub["diagnosis"] == "AD").astype(int).values)
    prob = model.predict(Xlb)
    pc = np.clip(prob, 1e-12, 1 - 1e-12)
    score = np.log(pc / (1 - pc))  # log-odds: same ranking (AUC unchanged) but usable in linear/logistic adjustment
    calib = {"median_predicted_prob": float(np.median(prob)), "max_predicted_prob": float(prob.max()),
             "frac_predicted_AD_at_0.5": float((prob > 0.5).mean()),
             "sensitivity_at_0.5": float(((prob > 0.5) & (ylb == 1)).sum() / (ylb == 1).sum()),
             "specificity_at_0.5": float(((prob <= 0.5) & (ylb == 0)).sum() / (ylb == 0).sum())}
    print(f"Calibration on lockbox: median predicted P(AD)={calib['median_predicted_prob']:.2e}, max={calib['max_predicted_prob']:.3f}, "
          f"sensitivity@0.5={calib['sensitivity_at_0.5']:.2f}, specificity@0.5={calib['specificity_at_0.5']:.2f}")
    main_auc, main_lo, main_hi = _boot_auc(ylb, score)
    perm_p, perm_ge = _perm_p_auc(ylb, score)
    overlap = len([g for g in model.selected_genes if g in Xlb.columns])
    print(f"Frozen H3 model: lockbox AUC {main_auc:.3f} [{main_lo:.3f}, {main_hi:.3f}]; permutation p={perm_p:.4g} ({perm_ge}/{N_PERM} perms >= observed); feature overlap {overlap}/{len(model.selected_genes)}")

    # ---- 2. age / sex only ----
    age_only_auc = calc_auc(ylb, lb["age"].values)
    ao = _boot_auc(ylb, lb["age"].values)
    clf = LogisticRegression(max_iter=1000).fit(tr[["age", "is_female"]].values, ytr)
    s_transfer = clf.predict_proba(lb[["age", "is_female"]].values)[:, 1]
    transfer_as = _boot_auc(ylb, s_transfer)
    cv_pred = np.zeros(len(lb))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(lb, ylb):
        c = LogisticRegression(max_iter=1000).fit(lb.iloc[a][["age", "is_female"]].values, ylb[a])
        cv_pred[b] = c.predict_proba(lb.iloc[b][["age", "is_female"]].values)[:, 1]
    cv_as = _boot_auc(ylb, cv_pred)
    print(f"Lockbox age-only AUC {age_only_auc:.3f} [{ao[1]:.3f},{ao[2]:.3f}]; age+sex transferred from GSE118553 {transfer_as[0]:.3f} [{transfer_as[1]:.3f},{transfer_as[2]:.3f}]; age+sex 5-fold CV inside lockbox {cv_as[0]:.3f} [{cv_as[1]:.3f},{cv_as[2]:.3f}]")

    # ---- 3. age/sex-residualised model (betas from training donors only) ----
    genes = [g for g in e118.index if g in e132.index]
    Xtr_g = e118[tr.index].loc[genes].T.values
    Xlb_g = e132[lb.index].loc[genes].T.values
    Xtr_r, Xlb_r = residualize_covariates(Xtr_g, tr[["age", "is_female"]].values, Xlb_g, lb[["age", "is_female"]].values)
    m_res = InFoldLinearModel(model_type="ridge_classifier", top_hvg=1000, seed=42)
    m_res.fit_transform_train(pd.DataFrame(Xtr_r, columns=genes), ytr)
    s_res = m_res.predict(pd.DataFrame(Xlb_r, columns=genes))
    resid_model = _boot_auc(ylb, s_res)
    print(f"Age+sex-residualised model on lockbox: AUC {resid_model[0]:.3f} [{resid_model[1]:.3f},{resid_model[2]:.3f}]")

    # ---- 4. score-level adjustment inside lockbox ----
    z = (score - score.mean()) / score.std()
    Xfull = sm.add_constant(np.column_stack([z, lb["age"].values, lb["is_female"].values]))
    Xred = sm.add_constant(np.column_stack([lb["age"].values, lb["is_female"].values]))
    lr_full = sm.Logit(ylb, Xfull).fit(disp=0)
    lr_red = sm.Logit(ylb, Xred).fit(disp=0)
    lr_stat = 2 * (lr_full.llf - lr_red.llf)
    p_lr = float(stats.chi2.sf(lr_stat, 1))
    auc_full = calc_auc(ylb, lr_full.predict(Xfull)); auc_red = calc_auc(ylb, lr_red.predict(Xred))
    or_score = float(np.exp(lr_full.params[1])); or_ci = np.exp(lr_full.conf_int()[1]).tolist()  # per SD of the log-odds score
    print(f"Lockbox logistic AD ~ score + age + sex: OR per SD of score {or_score:.2f} [{or_ci[0]:.2f},{or_ci[1]:.2f}], LR-test p={p_lr:.3g}; AUC age+sex {auc_red:.3f} -> +score {auc_full:.3f}")
    # score residualised on age+sex within the lockbox
    Zc = sm.add_constant(lb[["age", "is_female"]].values)
    score_res = score - Zc @ np.linalg.lstsq(Zc, score, rcond=None)[0]
    score_resid_auc = _boot_auc(ylb, score_res)
    print(f"AUC of the lockbox score after regressing out age+sex: {score_resid_auc[0]:.3f} [{score_resid_auc[1]:.3f},{score_resid_auc[2]:.3f}]")

    # ---- 5. age-matched subset ----
    pairs = _match_age(lb, caliper=None)
    keep_idx = [i for pr in pairs for i in pr[:2]]
    mlb = lb.loc[keep_idx]
    pos = [lb.index.get_loc(i) for i in keep_idx]
    y_m = ylb[pos]; s_m = score[pos]
    matched = _boot_auc(y_m, s_m)
    md = _describe(mlb, "age-matched lockbox subset")
    print(f"Age-matched lockbox subset ({len(pairs)} pairs, sex-exact): AD {md['age_AD_mean']:.1f} vs CTL {md['age_CTL_mean']:.1f} (p={md['age_diff_welch_p']:.3g}); AUC {matched[0]:.3f} [{matched[1]:.3f},{matched[2]:.3f}]")

    # ---- 5b. age strata ----
    strata = {}
    q = lb["age"].quantile([1 / 3, 2 / 3]).values
    for name, mask in [("youngest_tertile", lb["age"] <= q[0]), ("middle_tertile", (lb["age"] > q[0]) & (lb["age"] <= q[1])), ("oldest_tertile", lb["age"] > q[1])]:
        mk = mask.values
        if len(np.unique(ylb[mk])) == 2:
            strata[name] = {"n": int(mk.sum()), "auc": calc_auc(ylb[mk], score[mk]), "age_range": [float(lb["age"][mk].min()), float(lb["age"][mk].max())]}
    print("Age-tertile AUCs:", {k: round(v["auc"], 3) for k, v in strata.items()})

    res = {
        "analysis": "H3_age_sex_robustness",
        "model": {
            "algorithm": "L2-regularised logistic regression (sklearn LogisticRegressionCV, Cs=5, inner 3-fold CV on AUC)",
            "features": "top 1000 highest-variance genes in the training set, z-scored with training mean/SD",
            "training_cohort": "GSE118553",
            "training_regions": ["Temporal_Cortex"],
            "training_labels": "AD vs CTL (AsymAD excluded)",
            "n_train_samples": int(len(sub)), "n_train_donors": int(sub["donor_id"].nunique()),
            "n_features": int(len(model.selected_genes)), "lockbox_feature_overlap": overlap,
            "lockbox_touched_for_training_or_tuning": False,
        },
        "demographics_lockbox": demo_lb,
        "demographics_training": demo_tr,
        "lockbox_calibration": calib,
        "lockbox_main": {"auc": main_auc, "ci_lower": main_lo, "ci_upper": main_hi, "perm_p": perm_p, "n_perm_ge_observed": perm_ge, "n_perm": N_PERM,
                         "min_attainable_p": 1 / (N_PERM + 1)},
        "age_only_auc": {"auc": age_only_auc, "ci_lower": ao[1], "ci_upper": ao[2]},
        "age_sex_transfer_auc": {"auc": transfer_as[0], "ci_lower": transfer_as[1], "ci_upper": transfer_as[2]},
        "age_sex_within_lockbox_cv_auc": {"auc": cv_as[0], "ci_lower": cv_as[1], "ci_upper": cv_as[2]},
        "age_sex_residualised_model_auc": {"auc": resid_model[0], "ci_lower": resid_model[1], "ci_upper": resid_model[2]},
        "score_logistic_adjusted": {"or_per_sd": or_score, "or_ci_lower": or_ci[0], "or_ci_upper": or_ci[1], "lr_test_p": p_lr,
                                    "auc_age_sex": auc_red, "auc_age_sex_plus_score": auc_full},
        "score_residualised_on_age_sex_auc": {"auc": score_resid_auc[0], "ci_lower": score_resid_auc[1], "ci_upper": score_resid_auc[2]},
        "age_matched_subset": {"n_pairs": int(len(pairs)), "auc": matched[0], "ci_lower": matched[1], "ci_upper": matched[2],
                               "age_AD_mean": md["age_AD_mean"], "age_CTL_mean": md["age_CTL_mean"], "age_diff_welch_p": md["age_diff_welch_p"]},
        "age_tertile_auc": strata,
    }
    with open("results/h3_age_checks.json", "w") as f:
        json.dump(res, f, indent=2)

    log_analysis(
        analysis_id="H3_AGE_001",
        category="SENSITIVITY",
        hypothesis_or_idea="H3 lockbox age/sex robustness (post-review)",
        description="Age+sex-only AUC, age/sex-residualised model, score-adjusted logistic model and age-matched subset on GSE132903",
        cohorts_used="GSE118553 -> GSE132903",
        regions_analyzed="Temporal Cortex / MTG",
        model_type="InFold_LogisticCV_frozen_plus_covariate_controls",
        covariates_adjusted="Age, Sex",
        sample_size_donors=int(len(lb)),
        primary_metric="Age_sex_residualised_lockbox_AUC",
        point_estimate=resid_model[0],
        ci_95_lower=resid_model[1],
        ci_95_upper=resid_model[2],
        p_value=p_lr,
        replicated_in_lockbox=bool(resid_model[1] > 0.5),
        status="COMPLETE",
        notes=f"Main lockbox AUC {main_auc:.3f}; age+sex-only AUC {transfer_as[0]:.3f}; age-matched AUC {matched[0]:.3f}; p is the LR test for the score given age+sex.",
    )
    return res


if __name__ == "__main__":
    run_h3_age_checks()
