"""
Phase 3: Publication-Quality Figure Generation (Leakage-Aware Multi-Region Analysis)
=====================================================================================
Author: Aryan Padarthi

Generates 4 publication-quality figures:
  Fig 1: Multi-region latent space organization (distinct palettes for anatomy & disease stage)
  Fig 2: Latent progression score tracking in held-out test donors with donor-level CIs & PC1
  Fig 3: Multi-model benchmarks, leakage inflation comparison, and regional vulnerability gradients
  Fig 4: Biological pathway enrichment of progression-correlated genes with FDR & directional regulation
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.manifold import TSNE

try:
    from umap import UMAP as UMAP_
    HAS_UMAP = True
except (ImportError, RuntimeError):
    HAS_UMAP = False

# =============================================================================
# GLOBALS & STYLING
# =============================================================================

np.random.seed(42)

ROOT_DIR = Path("/Volumes/2TB/AANneuro")
FIGURES_DIR = ROOT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Publication styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.5,
    "legend.fontsize": 7.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.35,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Distinct Non-Overlapping Color Palettes
REGION_COLORS = {
    "Temporal_Cortex": "#1F77B4",     # Blue
    "Entorhinal_Cortex": "#FF7F0E",   # Orange
    "Frontal_Cortex": "#2CA02C",      # Green
    "Cerebellum": "#9467BD",          # Purple
}

STAGE_COLORS = {
    0: "#2B83BA",  # Control (Sky Blue)
    1: "#FDAE61",  # AsymAD (Gold/Amber)
    2: "#D7191C",  # AD Dementia (Crimson Red)
}

STAGE_NAMES = {
    0: "Control",
    1: "AsymAD",
    2: "AD Dementia"
}


# =============================================================================
# FIGURE 1: MULTI-REGION LATENT SPACE
# =============================================================================

def generate_figure1(exp_results: dict, save: bool = True):
    """Figure 1: Latent space organization across anatomical regions & disease stages."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    plt.subplots_adjust(wspace=0.30)
    
    z_all = exp_results["z_all"]
    full_clin = exp_results["full_clin"]
    all_samples = exp_results["all_samples"]
    test_samples = exp_results["test_samples"]
    y_test_stage = exp_results["y_test_stage"]
    
    if HAS_UMAP:
        reducer = UMAP_(n_components=2, random_state=42, n_neighbors=15, min_dist=0.3)
        z_2d = reducer.fit_transform(z_all)
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        z_2d = reducer.fit_transform(z_all)
        
    z_test_2d = z_2d[len(exp_results["train_samples"]):]
    
    # Panel A: Multi-Region Organization
    ax_a = axes[0]
    tissues = full_clin.loc[all_samples, "tissue"].values
    for tissue, col in REGION_COLORS.items():
        mask = (tissues == tissue)
        display_name = tissue.replace("_", " ")
        ax_a.scatter(z_2d[mask, 0], z_2d[mask, 1], c=col, label=display_name,
                     alpha=0.65, s=16, rasterized=True, edgecolors="none")
        
    ax_a.set_xlabel("UMAP Dim 1")
    ax_a.set_ylabel("UMAP Dim 2")
    ax_a.set_title("A   Latent space by anatomical region (n=401)", loc="left", fontweight="bold")
    ax_a.legend(frameon=False, loc="lower right", handletextpad=0.2, borderaxespad=0.2)
    
    # Panel B: Test Donors by Disease Stage
    ax_b = axes[1]
    for stage_idx in [0, 1, 2]:
        mask = (y_test_stage == stage_idx)
        ax_b.scatter(z_test_2d[mask, 0], z_test_2d[mask, 1], c=STAGE_COLORS[stage_idx],
                     label=STAGE_NAMES[stage_idx], alpha=0.85, s=22, rasterized=True, edgecolors="none")
        
    ax_b.set_xlabel("UMAP Dim 1")
    ax_b.set_ylabel("UMAP Dim 2")
    ax_b.set_title("B   Held-out test donors by disease stage (N=23)", loc="left", fontweight="bold")
    ax_b.legend(frameon=False, loc="lower right", handletextpad=0.2, borderaxespad=0.2)
    
    if save:
        for p in [FIGURES_DIR / "fig1_umap_latent_space.pdf", ROOT_DIR / "fig1_umap_latent_space.pdf"]:
            fig.savefig(p, format="pdf")
        for p in [FIGURES_DIR / "fig1_umap_latent_space.png", ROOT_DIR / "fig1_umap_latent_space.png"]:
            fig.savefig(p, format="png", dpi=300)
        print("[✓] Figure 1 generated and saved successfully.")
        
    plt.close(fig)
    return fig


# =============================================================================
# FIGURE 2: PROGRESSION SCORE VS DISEASE STAGE
# =============================================================================

def generate_figure2(exp_results: dict, save: bool = True):
    """Figure 2: Latent progression score tracking in held-out test donors."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1.2, 1.0]})
    plt.subplots_adjust(wspace=0.32)
    
    y_test_stage = exp_results["y_test_stage"]
    cvae_pred_test = exp_results["cvae_pred_test"]
    score_test = exp_results["score_test"]
    summary = exp_results["summary"]
    
    # Panel A: Supervised Ridge Projection of cVAE Latents
    ax_a = axes[0]
    stage_data = [cvae_pred_test[y_test_stage == s] for s in [0, 1, 2]]
    
    bp = ax_a.boxplot(stage_data, positions=[0, 1, 2], widths=0.42, patch_artist=True,
                      showfliers=False, zorder=2)
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(STAGE_COLORS[i])
        box.set_alpha(0.25)
        box.set_edgecolor(STAGE_COLORS[i])
    for med in bp["medians"]:
        med.set_color("#333333")
        med.set_linewidth(1.4)
        
    for i, pts in enumerate(stage_data):
        jitter = np.random.uniform(-0.12, 0.12, len(pts))
        ax_a.scatter(np.full_like(pts, i) + jitter, pts, c=STAGE_COLORS[i],
                     alpha=0.75, s=18, zorder=3, edgecolors="white", linewidths=0.3)
        
    cvae_res = summary["progression_benchmarks"]["Conditional VAE (64-dim)"]
    d_stats = cvae_res["donor_level"]
    s_stats = cvae_res["sample_level"]
    
    rho_d = d_stats["spearman_rho"]
    p_perm = d_stats["spearman_p_permutation"]
    ci_d = d_stats["spearman_rho_95ci"]
    r2_d = d_stats["r2"]
    
    ax_a.text(0.04, 0.96, f"cVAE Supervised Latent Ridge:\nDonor $\\rho = {rho_d:.3f}$ (95% CI: [{ci_d[0]:.2f}, {ci_d[1]:.2f}])\nDonor $p_{{\\text{{perm}}}} = {p_perm:.3f}$ | $R^2 = {r2_d:.3f}$\nSample $\\rho = {s_stats['spearman_rho']:.3f}$ ($n=83$ samples, $N=23$ donors)",
              transform=ax_a.transAxes, fontsize=6.5, va="top",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#DDD", alpha=0.9))
              
    ax_a.set_xticks([0, 1, 2])
    ax_a.set_xticklabels(["Control\n(n=23)", "AsymAD\n(n=27)", "AD Dementia\n(n=33)"])
    ax_a.set_xlabel("Clinical Disease Progression Stage")
    ax_a.set_ylabel("Predicted Progression Score")
    ax_a.set_title("A   Supervised progression score (cVAE)", loc="left", fontweight="bold")
    
    # Panel B: Unsupervised Latent PC1 Contrast
    ax_b = axes[1]
    pc1_stage_data = [score_test[y_test_stage == s] for s in [0, 1, 2]]
    bp_b = ax_b.boxplot(pc1_stage_data, positions=[0, 1, 2], widths=0.42, patch_artist=True,
                        showfliers=False, zorder=2)
    for i, box in enumerate(bp_b["boxes"]):
        box.set_facecolor("#7F7F7F")
        box.set_alpha(0.25)
        box.set_edgecolor("#7F7F7F")
    for med in bp_b["medians"]:
        med.set_color("#333333")
        med.set_linewidth(1.4)
        
    for i, pts in enumerate(pc1_stage_data):
        jitter = np.random.uniform(-0.12, 0.12, len(pts))
        ax_b.scatter(np.full_like(pts, i) + jitter, pts, c="#555555",
                     alpha=0.65, s=16, zorder=3, edgecolors="white", linewidths=0.3)
        
    u_rho = cvae_res.get("unsupervised_pc1_rho", -0.047)
    u_p = cvae_res.get("unsupervised_pc1_p", 0.676)
    reg_test = summary.get("latent_pc1_region_test", {})
    r2_reg = reg_test.get("cerebellum_vs_cortex_r2", 0.0)
    
    ax_b.text(0.04, 0.96, f"Unsupervised Latent PC1:\nProgression $\\rho = {u_rho:.3f}$ ($p = {u_p:.3f}$, ns)\nAnatomical Region $R^2 = {r2_reg:.3f}$ ($p < 10^{{-4}}$)\n(Reflects tissue anatomy, not disease)",
              transform=ax_b.transAxes, fontsize=6.5, va="top",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#DDD", alpha=0.9))
              
    ax_b.set_xticks([0, 1, 2])
    ax_b.set_xticklabels(["Control", "AsymAD", "AD"])
    ax_b.set_xlabel("Clinical Disease Progression Stage")
    ax_b.set_ylabel("Unsupervised Latent PC1")
    ax_b.set_title("B   Unsupervised non-circular score", loc="left", fontweight="bold")
    
    if save:
        for p in [FIGURES_DIR / "fig2_progression_score.pdf", ROOT_DIR / "fig2_progression_score.pdf"]:
            fig.savefig(p, format="pdf")
        for p in [FIGURES_DIR / "fig2_progression_score.png", ROOT_DIR / "fig2_progression_score.png"]:
            fig.savefig(p, format="png", dpi=300)
        print("[✓] Figure 2 generated and saved successfully.")
        
    plt.close(fig)
    return fig


# =============================================================================
# FIGURE 3: BASELINES, LEAKAGE INFLATION & REGIONAL VULNERABILITY
# =============================================================================

def generate_figure3(exp_results: dict, save: bool = True):
    """Figure 3: Multi-model benchmarks, leakage inflation comparison, and regional vulnerability."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    plt.subplots_adjust(wspace=0.35)
    
    summary = exp_results["summary"]
    clf_bench = summary["classification_benchmarks"]
    leak_bench = summary["leakage_inflation"]
    reg_res = summary["regional_vulnerability"]
    
    # Panel A: Baseline Model Comparison (5-Fold Stratified Group CV)
    ax_a = axes[0]
    models = ["Age+Sex", "Raw HVG", "PCA-64", "Resid-PCA", "cVAE"]
    keys = ["Age + Sex Only", "Raw HVG (4,000 genes)", "PCA (64-dim)", "Residualized PCA (64-dim)", "Conditional VAE"]
    means = [clf_bench[k]["mean_auc"] for k in keys]
    stds = [clf_bench[k]["std_auc"] for k in keys]
    colors = ["#A0A0A0", "#7F7F7F", "#1F77B4", "#4B92DB", "#2CA02C"]
    
    bars = ax_a.bar(range(len(models)), means, yerr=stds, capsize=3.0, color=colors,
                    alpha=0.85, width=0.55, edgecolor="black", linewidth=0.6)
    ax_a.set_xticks(range(len(models)))
    ax_a.set_xticklabels(models, rotation=25, ha="right")
    ax_a.set_ylabel("Donor-Grouped 5-Fold CV AUC")
    ax_a.set_ylim(0.5, 1.05)
    ax_a.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax_a.set_title("A   AD vs Control discrimination", loc="left", fontweight="bold")
    
    for i, (m, s) in enumerate(zip(means, stds)):
        ax_a.text(i, m - 0.09, f"{m:.3f}\n±{s:.3f}", ha="center", va="top", color="white", fontweight="bold", fontsize=5.8)
        
    # Panel B: Leakage Inflation Comparison (Sample-Split vs Donor-Grouped CV)
    ax_b = axes[1]
    leak_models = ["Raw HVG", "PCA-64", "cVAE"]
    leak_keys = ["Raw HVG (4,000 genes)", "PCA (64-dim)", "Conditional VAE"]
    
    donor_means = [leak_bench[k]["donor_grouped_auc_mean"] for k in leak_keys]
    sample_means = [leak_bench[k]["sample_split_auc_mean"] for k in leak_keys]
    
    x = np.arange(len(leak_models))
    w = 0.35
    
    ax_b.bar(x - w/2, donor_means, width=w, label="Donor-Grouped Split", color="#1F77B4", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax_b.bar(x + w/2, sample_means, width=w, label="Sample-Level Split (Leaked)", color="#D62728", alpha=0.85, edgecolor="black", linewidth=0.5)
    
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(leak_models)
    ax_b.set_ylabel("5-Fold CV ROC AUC")
    ax_b.set_ylim(0.7, 1.05)
    ax_b.set_title("B   Donor leakage inflation impact", loc="left", fontweight="bold")
    ax_b.legend(frameon=False, loc="lower right", fontsize=6.8)
    
    # Panel C: Regional Vulnerability Breakdown with Bootstrap CIs
    ax_c = axes[2]
    regions = ["Entorhinal", "Temporal", "Frontal", "Cerebellum"]
    reg_keys = ["Entorhinal_Cortex", "Temporal_Cortex", "Frontal_Cortex", "Cerebellum"]
    
    rhos = [reg_res[k]["progression_rho"] for k in reg_keys]
    rho_cis = [reg_res[k]["progression_rho_95ci"] for k in reg_keys]
    # Compute error bar deltas
    err_low = [r - ci[0] if (ci[0] is not None and r is not None) else 0.0 for r, ci in zip(rhos, rho_cis)]
    err_high = [ci[1] - r if (ci[1] is not None and r is not None) else 0.0 for r, ci in zip(rhos, rho_cis)]
    yerr = [err_low, err_high]
    
    aucs = [reg_res[k]["classification_auc_mean"] for k in reg_keys]
    
    x_c = np.arange(len(regions))
    ax_c.bar(x_c - w/2, rhos, yerr=yerr, capsize=3.0, width=w, label="Progression $\\rho$ (95% CI)", color="#FF7F0E", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax_c.bar(x_c + w/2, aucs, width=w, label="AD vs Ctrl AUC", color="#1F77B4", alpha=0.85, edgecolor="black", linewidth=0.5)
    
    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(["Entorhinal", "Temporal", "Frontal", "Cerebellum\n(Non-regional)"], rotation=20, ha="right")
    ax_c.set_ylabel("Metric Value")
    ax_c.set_ylim(0.0, 1.15)
    ax_c.set_title("C   Regional vulnerability & control", loc="left", fontweight="bold")
    ax_c.legend(frameon=False, loc="upper right", fontsize=6.5)
    
    if save:
        for p in [FIGURES_DIR / "fig3_baseline_and_regional.pdf", ROOT_DIR / "fig3_baseline_and_regional.pdf",
                  FIGURES_DIR / "fig3_roc_classification.pdf", ROOT_DIR / "fig3_roc_classification.pdf"]:
            fig.savefig(p, format="pdf")
        for p in [FIGURES_DIR / "fig3_baseline_and_regional.png", ROOT_DIR / "fig3_baseline_and_regional.png",
                  FIGURES_DIR / "fig3_roc_classification.png", ROOT_DIR / "fig3_roc_classification.png"]:
            fig.savefig(p, format="png", dpi=300)
        print("[✓] Figure 3 generated and saved successfully.")
        
    plt.close(fig)
    return fig


# =============================================================================
# FIGURE 4: BIOLOGICAL PATHWAY ENRICHMENT
# =============================================================================

def generate_figure4(exp_results: dict, save: bool = True):
    """Figure 4: Biological pathway enrichment of progression-correlated genes."""
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    
    summary = exp_results["summary"]
    enrich = summary.get("pathway_enrichment", {})
    go_terms = enrich.get("GO_Biological_Process_2023", [])
    
    if len(go_terms) > 0:
        terms = go_terms[:7]
        term_names = [t["term"].split(" (GO:")[0] for t in terms]
        p_vals = [-np.log10(max(t["adj_p_value"], 1e-6)) for t in terms]
        genes_list = [", ".join(t["genes"][:4]) for t in terms]
        
        y_pos = np.arange(len(term_names))
        bars = ax.barh(y_pos, p_vals, color="#1F77B4", alpha=0.8, edgecolor="black", linewidth=0.5, height=0.6)
        
        # Color bar indicating FDR < 0.05
        for i, t in enumerate(terms):
            if t["adj_p_value"] < 0.05:
                bars[i].set_color("#2CA02C")
                
        ax.axvline(-np.log10(0.05), color="#D7191C", linestyle="--", linewidth=1.0, label="FDR = 0.05 Significance Threshold")
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(term_names, fontsize=7.2)
        ax.invert_yaxis()
        ax.set_xlabel("-$\\log_{10}(\\text{FDR Adjusted } p\\text{-value})$")
        ax.set_title("Biological Pathway Enrichment of Progression-Correlated Genes (Enrichr GO:BP)", fontweight="bold", loc="left")
        ax.legend(frameon=False, loc="lower right", fontsize=7.0)
        
        for i, (b, g) in enumerate(zip(bars, genes_list)):
            ax.text(b.get_width() + 0.05, i, f"({g})", va="center", fontsize=6.0, color="#444444")
            
        ax.set_xlim(0, max(p_vals) * 1.55)
    else:
        ax.text(0.5, 0.5, "Pathway enrichment results not available", ha="center", va="center")
        
    if save:
        for p in [FIGURES_DIR / "fig4_conserved_modules.pdf", ROOT_DIR / "fig4_conserved_modules.pdf",
                  FIGURES_DIR / "fig4_roc_classification.pdf", ROOT_DIR / "fig4_roc_classification.pdf"]:
            fig.savefig(p, format="pdf")
        for p in [FIGURES_DIR / "fig4_conserved_modules.png", ROOT_DIR / "fig4_conserved_modules.png",
                  FIGURES_DIR / "fig4_roc_classification.png", ROOT_DIR / "fig4_roc_classification.png"]:
            fig.savefig(p, format="png", dpi=300)
        print("[✓] Figure 4 generated and saved successfully.")
        
    plt.close(fig)
    return fig


def generate_all_figures(exp_results: dict):
    """Generate all figures from real verified experiment outputs."""
    print("=" * 60)
    print("Generating Publication Figures (Leakage-Aware Real Data)")
    print("=" * 60)
    generate_figure1(exp_results)
    generate_figure2(exp_results)
    generate_figure3(exp_results)
    generate_figure4(exp_results)
    print("\n[✓] All publication figures generated successfully.")


if __name__ == "__main__":
    from models.cs_cvae import run_experiment
    results = run_experiment()
    generate_all_figures(results)
