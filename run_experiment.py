"""
Full Experiment Runner (Verified Donor-Grouped Pipeline)
========================================================
Executes the verified end-to-end pipeline:
  1. Donor-grouped preprocessing & train-isolated HVG selection
  2. Model training (cVAE, VAE, PCA, Raw HVG baselines)
  3. Strict donor-grouped evaluation & regional vulnerability analysis
  4. Biological pathway enrichment via Enrichr
  5. Publication figure generation

Run: python run_experiment.py
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing.data_acquisition import run_preprocessing_pipeline
from models.cs_cvae import run_experiment
from figures.generate_figures import generate_all_figures


def main():
    """Execute the verified experiment pipeline."""
    
    print("\n" + "█" * 75)
    print("█  AAN Neuroscience Research Prize — Donor-Grouped Verified Pipeline     █")
    print("█  Author: Aryan Padarthi                                                █")
    print("█" * 75 + "\n")
    
    # Phase 2: Preprocessing
    print("\n" + "─" * 60)
    print("PHASE 2: Data Verification & Donor-Grouped Preprocessing")
    print("─" * 60)
    run_preprocessing_pipeline()
    
    # Phase 3A: Model Training & Evaluation
    print("\n" + "─" * 60)
    print("PHASE 3A: Model Training, Baselines & Regional Evaluation")
    print("─" * 60)
    exp_results = run_experiment()
    
    # Phase 3B: Figure Generation
    print("\n" + "─" * 60)
    print("PHASE 3B: Publication Figure Generation")
    print("─" * 60)
    generate_all_figures(exp_results=exp_results)
    
    print("\n" + "=" * 65)
    print("EXPERIMENT PIPELINE COMPLETE")
    print("=" * 65)
    
    summary = exp_results.get("summary", {})
    pb = summary.get("progression_benchmarks", {})
    cb = summary.get("classification_benchmarks", {})
    rv = summary.get("regional_vulnerability", {})
    resil = summary.get("resilience_asymad_vs_ad", {})
    leak = summary.get("leakage_inflation", {})
    
    print("\n--- Summary of Verified Results ---")
    print(f"Cohort: {summary['cohort']['n_donors_total']} donors ({summary['cohort']['n_samples_total']} samples)")
    print(f"  Train: {summary['cohort']['n_donors_train']} donors | Test: {summary['cohort']['n_donors_test']} donors (0% overlap)")
    
    print("\nProgression Tracking in Held-Out Test Donors (Donor-Level Aggregated vs Sample-Level):")
    for k, v in pb.items():
        d = v.get("donor_level", {})
        s = v.get("sample_level", {})
        ci = d.get("spearman_rho_95ci", [0, 0])
        alpha = v.get("tuned_alpha", 1.0)
        print(f"  {k:28s} (α={alpha:6.1f}): Donor ρ = {d.get('spearman_rho', 0):.3f} (95% CI: [{ci[0]:.2f}, {ci[1]:.2f}], p_perm={d.get('spearman_p_permutation', 0):.3f}, R² = {d.get('r2', 0):.3f}) | Sample ρ = {s.get('spearman_rho', 0):.3f}")
        
    print("\nDonor-Grouped 5-Fold CV Discrimination (AD vs Control & AsymAD vs AD):")
    for k, v in cb.items():
        asym_auc = resil.get(k, {}).get("mean_auc", 0.0)
        asym_std = resil.get(k, {}).get("std_auc", 0.0)
        print(f"  {k:28s}: AD vs Ctrl AUC = {v['mean_auc']:.4f} ± {v['std_auc']:.4f} | AsymAD vs AD AUC = {asym_auc:.4f} ± {asym_std:.4f}")
        
    print("\nDonor Leakage Inflation Comparison (Sample-Split vs Donor-Grouped Split):")
    for k, v in leak.items():
        print(f"  {k:28s}: Donor Split AUC = {v['donor_grouped_auc_mean']:.4f} | Leaked Sample Split AUC = {v['sample_split_auc_mean']:.4f} (Δ = {v['inflation_delta']:+.4f})")
        
    print("\nRegional Vulnerability Gradients & Cerebellar Non-Zero Signal:")
    for reg, stats in rv.items():
        ci = stats.get("progression_rho_95ci", [0, 0])
        print(f"  {reg:22s}: Progression ρ = {stats['progression_rho']:.3f} (95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]), AD vs Ctrl AUC = {stats['classification_auc_mean']:.3f} ± {stats['classification_auc_std']:.3f}")
        
    print(f"\nOutput files:")
    print(f"  Figures: /Volumes/2TB/AANneuro/figures/fig[1-4]*.pdf|png")
    print(f"  Results: /Volumes/2TB/AANneuro/results/summary_results.json")
    print(f"  Data:    /Volumes/2TB/AANneuro/data/processed/")


if __name__ == "__main__":
    main()
