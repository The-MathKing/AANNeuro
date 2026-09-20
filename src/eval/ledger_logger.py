"""
Analysis Ledger Logger.
Appends every executed experiment, hypothesis test, and exploratory benchmark to ledger.csv.
"""

import os
import datetime
import pandas as pd


def log_analysis(
    analysis_id,
    category,
    hypothesis_or_idea,
    description,
    cohorts_used,
    regions_analyzed,
    model_type,
    covariates_adjusted,
    sample_size_donors,
    primary_metric,
    point_estimate,
    ci_95_lower,
    ci_95_upper,
    p_value,
    replicated_in_lockbox,
    status,
    notes="",
    ledger_path="ledger.csv"
):
    """
    Log an analysis record to ledger.csv.
    """
    record = {
        "analysis_id": analysis_id,
        "category": category,
        "hypothesis_or_idea": hypothesis_or_idea,
        "description": description,
        "cohorts_used": cohorts_used,
        "regions_analyzed": regions_analyzed,
        "model_type": model_type,
        "covariates_adjusted": covariates_adjusted,
        "sample_size_donors": sample_size_donors,
        "primary_metric": primary_metric,
        "point_estimate": f"{point_estimate:.4f}" if isinstance(point_estimate, (float, int)) and not pd.isna(point_estimate) else str(point_estimate),
        "ci_95_lower": f"{ci_95_lower:.4f}" if isinstance(ci_95_lower, (float, int)) and not pd.isna(ci_95_lower) else str(ci_95_lower),
        "ci_95_upper": f"{ci_95_upper:.4f}" if isinstance(ci_95_upper, (float, int)) and not pd.isna(ci_95_upper) else str(ci_95_upper),
        "p_value": f"{p_value:.4e}" if isinstance(p_value, (float, int)) and not pd.isna(p_value) else str(p_value),
        "replicated_in_lockbox": str(replicated_in_lockbox),
        "status": status,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "notes": notes
    }

    df_record = pd.DataFrame([record])
    if os.path.exists(ledger_path) and os.path.getsize(ledger_path) > 0:
        df_record.to_csv(ledger_path, mode="a", header=False, index=False)
    else:
        df_record.to_csv(ledger_path, mode="w", header=True, index=False)
    print(f"Logged to {ledger_path}: [{analysis_id}] {hypothesis_or_idea} - {primary_metric}={point_estimate}")
