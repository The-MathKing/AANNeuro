"""
Exploratory Analysis E6: Transcriptomic Brain Age Acceleration.
1. Trains a transcriptomic clock on neurologically normal controls (CDR=0 / CTL) using donor-grouped CV.
2. Predicts biological transcriptomic brain age in unseen test donors.
3. Computes Brain Age Gap / Acceleration: Delta_Age = Predicted_Age - Chronological_Age.
4. Tests whether cognitively resilient donors exhibit lower biological brain age acceleration compared to demented donors at matched pathology.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from src.load import load_gse84422, load_gse118553
from src.eval.cross_val import get_donor_grouped_splits
from src.eval.metrics import calc_pearson, calc_spearman
from src.eval.bootstrap import donor_bootstrap_ci
from src.eval.ledger_logger import log_analysis
from src.models.linear_pipeline import InFoldLinearModel


def run_e6_analysis():
    print("=== RUNNING EXPLORATORY ANALYSIS E6: TRANSCRIPTOMIC BRAIN AGE ===")
    os.makedirs("results", exist_ok=True)

    meta_msbb, expr_msbb = load_gse84422()
    # Filter to donors with valid age and donor_id
    valid = meta_msbb.dropna(subset=["age", "cdr", "braak", "donor_id"]).copy()
    valid["age"] = valid["age"].astype(float)
    valid["cdr"] = valid["cdr"].astype(float)
    valid["braak"] = valid["braak"].astype(float)

    # Use Frontal/Temporal cortical regions
    cortical = valid[valid["region"].isin(["BM9_DLPFC", "BM10_FP", "BM22_STG", "BM20_ITG", "BM36_PHG"])].copy()
    X_df = expr_msbb[cortical.index].T  # samples x genes

    # 1. Identify Controls (CDR = 0 and Braak <= 2)
    ctl_mask = (cortical["cdr"] == 0) & (cortical["braak"] <= 2)
    ctl_meta = cortical[ctl_mask]
    ctl_X = X_df.loc[ctl_meta.index]
    ctl_y_age = ctl_meta["age"].values
    ctl_donors = ctl_meta["donor_id"].values

    print(f"Controls for Brain Age Clock Training: {len(ctl_meta)} samples across {ctl_meta['donor_id'].nunique()} donors.")

    # 2. Donor-Grouped CV evaluation of the Age Clock on Controls
    splits_ctl = get_donor_grouped_splits(ctl_donors, n_splits=4, seed=42)
    ctl_pred_cv = np.full(len(ctl_meta), np.nan)
    
    for tr, te in splits_ctl:
        model = InFoldLinearModel(model_type="ridge_regression", top_hvg=500, seed=42)
        model.fit_transform_train(ctl_X.iloc[tr], ctl_y_age[tr])
        ctl_pred_cv[te] = model.predict(ctl_X.iloc[te])

    r_age_ctl, p_age_ctl = calc_pearson(ctl_y_age, ctl_pred_cv)
    print(f"Control Brain Age Clock CV Accuracy: Pearson r = {r_age_ctl:.3f} (p = {p_age_ctl:.2e})")

    # 3. Fit Final Clock on All Controls
    final_clock = InFoldLinearModel(model_type="ridge_regression", top_hvg=500, seed=42)
    final_clock.fit_transform_train(ctl_X, ctl_y_age)

    # 4. Predict Transcriptomic Age across all cortical samples
    cortical["pred_brain_age"] = final_clock.predict(X_df)
    cortical["delta_age"] = cortical["pred_brain_age"] - cortical["age"]

    # 5. Evaluate High-Pathology Cohort (Braak >= 4): Resilient (CDR <= 0.5) vs Demented (CDR >= 1.0)
    high_path = cortical[cortical["braak"] >= 4].copy()
    resilient_mask = high_path["cdr"] <= 0.5
    demented_mask = high_path["cdr"] >= 1.0

    delta_age_resil = high_path.loc[resilient_mask, "delta_age"].values
    delta_age_dem = high_path.loc[demented_mask, "delta_age"].values

    mean_delta_resil = float(np.mean(delta_age_resil))
    mean_delta_dem = float(np.mean(delta_age_dem))

    # T-test for age acceleration
    t_stat, p_val_ttest = stats.ttest_ind(delta_age_resil, delta_age_dem, equal_var=False)

    print(f"\nBrain Age Acceleration in High Neuropathology (Braak >= 4):")
    print(f"  Resilient Delta Age: {mean_delta_resil:+.2f} years (N = {len(delta_age_resil)})")
    print(f"  Demented Delta Age:  {mean_delta_dem:+.2f} years (N = {len(delta_age_dem)})")
    print(f"  Difference:          {mean_delta_dem - mean_delta_resil:.2f} years older in demented (p = {p_val_ttest:.4f})")

    results = {
        "analysis": "E6_Transcriptomic_Brain_Age",
        "clock_cv_pearson_r": float(r_age_ctl),
        "clock_cv_pearson_p": float(p_age_ctl),
        "resilient_delta_age_mean": mean_delta_resil,
        "demented_delta_age_mean": mean_delta_dem,
        "delta_age_difference": float(mean_delta_dem - mean_delta_resil),
        "t_statistic": float(t_stat),
        "p_value": float(p_val_ttest),
        "clock_valid": bool(r_age_ctl > 0.3),
        "conclusion": ("Clock failed validation (cross-validated r <= 0.3); the acceleration comparison is uninterpretable." if r_age_ctl <= 0.3 else "Clock validated; acceleration comparison reported as-is.")
    }

    with open("results/e6_brain_age.json", "w") as f:
        json.dump(results, f, indent=2)

    log_analysis(
        analysis_id="E6_AGE_001",
        category="EXPLORATORY",
        hypothesis_or_idea="E6 Transcriptomic Brain Age Acceleration",
        description="Brain age predictor fit on controls and evaluated on high-pathology resilient vs demented donors",
        cohorts_used="GSE84422_MSBB",
        regions_analyzed="Cortical association regions",
        model_type="Ridge_Brain_Age_Clock",
        covariates_adjusted="Chronological Age, Braak Stage (>=4)",
        sample_size_donors=cortical["donor_id"].nunique(),
        primary_metric="Delta_Age_Diff_Years",
        point_estimate=mean_delta_dem - mean_delta_resil,
        ci_95_lower=float(mean_delta_dem - mean_delta_resil - 1.96 * np.std(delta_age_dem)/np.sqrt(len(delta_age_dem))),
        ci_95_upper=float(mean_delta_dem - mean_delta_resil + 1.96 * np.std(delta_age_dem)/np.sqrt(len(delta_age_dem))),
        p_value=p_val_ttest,
        replicated_in_lockbox=False,
        status="COMPLETE",
        notes=f"Resilient Delta={mean_delta_resil:+.2f}y vs Demented Delta={mean_delta_dem:+.2f}y (diff={mean_delta_dem - mean_delta_resil:.2f}y, p={p_val_ttest:.4f})."
    )

    return results


if __name__ == "__main__":
    run_e6_analysis()
