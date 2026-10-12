"""
Master Analysis Execution Script.
Executes all primary pre-registered hypotheses (H1-H3) and all exploratory benchmarks (E1-E10 + Pan-Cortical).
Outputs JSON statistics into results/ and appends every experiment to ledger.csv.
"""

import os
import sys
import time

from src.analyses.h1_regional import run_h1_analysis
from src.analyses.h2_resilience import run_h2_analysis
from src.analyses.h3_replication import run_h3_analysis
from src.analyses.e1_leakage import run_e1_analysis
from src.analyses.e2_celltype import run_e2_analysis
from src.analyses.e3_variance import run_e3_analysis
from src.analyses.e4_dissociation import run_e4_analysis
from src.analyses.e5_biology import run_e5_analysis
from src.analyses.e6_brain_age import run_e6_analysis
from src.analyses.e7_transfer_matrix import run_e7_analysis
from src.analyses.e8_sparse_panel import run_e8_analysis
from src.analyses.e9_age_sensitivity import run_e9_analysis
from src.analyses.e10_cvae import run_e10_analysis
from src.analyses.h3_age_checks import run_h3_age_checks
from src.analyses.h3_cv_artifact_check import run_h3_cv_artifact_check
from src.analyses.h3_msbb_internal import run_h3_msbb_internal
from src.analyses.h1_robust import run_h1_robust
from src.analyses.h3_composition import run_h3_composition
from src.analyses.r5_checks import run_r5_checks
from src.analyses.resilience_overlap_donor import run_resilience_overlap_donor
from src.analyses.h2_donor_level import run_h2_donor_level
from src.analyses.h3_missing_features import run_h3_missing_features
from src.analyses.resilience_overlap import run_resilience_overlap
from src.analyses.pan_cortical_modeling import run_pan_cortical_analysis


def main():
    start_time = time.time()
    print("=" * 70)
    print("STARTING FULL AAN REPRODUCIBLE ANALYSIS PIPELINE")
    print("=" * 70)

    # 1. Primary Hypotheses
    print("\n[1/16] Executing H1 (Regional Vulnerability)...")
    run_h1_analysis()

    print("\n[2/16] Executing H2 (Cognitive Resilience)...")
    run_h2_analysis()

    print("\n[3/16] Executing H3 (External Replication in Lockbox)...")
    run_h3_analysis()

    # 2. Exploratory Analyses (Tier A: Must-Do)
    print("\n[4/16] Executing E1 (Leakage Taxonomy Benchmark)...")
    run_e1_analysis()

    print("\n[5/16] Executing E2 (Cell-Type Deconvolution & Vascular Integrity)...")
    run_e2_analysis()

    print("\n[6/16] Executing E3 (Linear Mixed Model Variance Partitioning)...")
    run_e3_analysis()

    print("\n[7/16] Executing E4 (Pathology vs Cognition Dissociation)...")
    run_e4_analysis()

    print("\n[8/16] Executing E5 (Biological Anchoring: AMP-AD & GWAS)...")
    run_e5_analysis()

    # 3. Exploratory Analyses (Tier B & C)
    print("\n[9/16] Executing E6 (Transcriptomic Brain Age Acceleration)...")
    run_e6_analysis()

    print("\n[10/16] Executing E7 (Cross-Region Transfer Matrix)...")
    run_e7_analysis()

    print("\n[11/16] Executing E8 (Sparse Stability-Selected Panel)...")
    run_e8_analysis()

    print("\n[12/16] Executing E9 (Age-Matched Sensitivity & Cerebellar Dissection)...")
    run_e9_analysis()

    print("\n[13/16] Executing E10 (cVAE vs Residualized Linear Benchmark)...")
    run_e10_analysis()

    print("\n[14/16] H3 age/sex robustness on the lockbox...")
    run_h3_age_checks()

    print("\n[15/16] Post-hoc resilience overlap (E4 genes vs GSE118553 AsymAD-vs-AD)...")
    run_resilience_overlap()

    print("\n[16/16] Pan-Cortical Multi-Region Training & Biological Prior Extension...")
    run_pan_cortical_analysis()

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"ALL 16 ANALYSES SUCCESSFULLY COMPLETED in {elapsed/60:.2f} minutes!")
    print("=" * 70)


if __name__ == "__main__":
    main()
