"""
Publication-Quality Figure Generation Suite (Figures 1-5).
Generates high-resolution vector PDF (for LaTeX publication) and 300 DPI PNGs.
Adheres to strict visual design principles: cohesive palette, clear panel labeling (A-D), 
explicit 95% bootstrap confidence intervals, and no chartjunk.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# Global styling configuration
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.6,
    'grid.alpha': 0.35,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

COLORS = {
    'primary': '#1f3b73',      # Deep Navy
    'secondary': '#008080',    # Teal
    'accent_red': '#d9534f',    # Coral Red
    'accent_green': '#2e7d32',  # Emerald Green
    'accent_gold': '#f0ad4e',   # Warm Amber
    'gray_dark': '#2c3e50',
    'gray_light': '#ecf0f1',
    'purple': '#6f42c1',
    'tiers': ['#c0392b', '#d35400', '#2980b9', '#27ae60', '#7f8c8d']
}


def load_json(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}


def generate_figure1_study_design():
    """Figure 1: Study Design, Multi-Cohort Architecture & Leakage-Aware Pipeline."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.axis('off')

    # Draw conceptual diagram boxes
    # Discovery Cohort
    bbox_props_disc = dict(boxstyle="round,pad=0.6", fc="#e8f4f8", ec="#2980b9", lw=1.5)
    bbox_props_msbb = dict(boxstyle="round,pad=0.6", fc="#fef5e7", ec="#d35400", lw=1.5)
    bbox_props_lock = dict(boxstyle="round,pad=0.6", fc="#eafaf1", ec="#27ae60", lw=1.5)
    bbox_props_infold = dict(boxstyle="round,pad=0.6", fc="#f4ecf7", ec="#8e44ad", lw=1.5)

    ax.text(0.18, 0.78, "Discovery Cohort\nGSE118553 (Patel et al.)\n4 Brain Regions (TC, EC, FC, CRB)\nN = 401 samples (112 donors)\nIllumina HumanHT-12 V4.0", 
            ha="center", va="center", bbox=bbox_props_disc, fontsize=9.5)

    ax.text(0.50, 0.78, "Primary Pathology & Resilience\nGSE84422 (Mount Sinai Brain Bank)\n19 Brain Regions\nN = 1,053 samples (118 donors)\nCDR (0-5), Braak (0-VI), CERAD", 
            ha="center", va="center", bbox=bbox_props_msbb, fontsize=9.5)

    ax.text(0.82, 0.78, "Strict Lockbox Validation\nGSE132903 (Piras et al.)\nMiddle Temporal Gyrus\nN = 195 samples (97 AD, 98 CTL)\nNot used for training/tuning;\nfrozen-model test, then post hoc controls", 
            ha="center", va="center", bbox=bbox_props_lock, fontsize=9.5)

    ax.text(0.50, 0.40, "Leakage-Aware Evaluation Engine\n- Donor-Grouped CV (Zero Donor Overlap)\n- In-Fold Feature Selection (Top HVGs fit on Train only)\n- In-Fold Covariate Residualization (Age, Sex)\n- Donor-Cluster Bootstrap (B = 2,000) & Permutations (N = 5,000)", 
            ha="center", va="center", bbox=bbox_props_infold, fontsize=10, weight="bold")

    # Outputs
    bbox_out = dict(boxstyle="round,pad=0.4", fc="#ffffff", ec="#2c3e50", lw=1.2)
    ax.text(0.18, 0.12, "H1: 19-Region Vulnerability\nRank correlation vs Braak order\n(permutation p < 0.05, Holm across H1, H3)", ha="center", va="center", bbox=bbox_out, fontsize=9)
    ax.text(0.50, 0.12, "H2: Cognitive Resilience (exploratory)\nBraak >= IV: CDR <= 0.5 vs CDR >= 1.0\nCross-cohort signature transfer", ha="center", va="center", bbox=bbox_out, fontsize=9)
    ax.text(0.82, 0.12, "H3: Lockbox Replication\nZero-leakage generalization\n(External AUC > 0.50)", ha="center", va="center", bbox=bbox_out, fontsize=9)

    # Arrows
    arrow_props = dict(facecolor='#2c3e50', edgecolor='#2c3e50', width=1.2, headwidth=6, shrink=0.08)
    ax.annotate('', xy=(0.35, 0.52), xytext=(0.20, 0.66), arrowprops=arrow_props)
    ax.annotate('', xy=(0.50, 0.52), xytext=(0.50, 0.66), arrowprops=arrow_props)
    ax.annotate('', xy=(0.65, 0.52), xytext=(0.80, 0.66), arrowprops=arrow_props)

    ax.annotate('', xy=(0.20, 0.20), xytext=(0.35, 0.30), arrowprops=arrow_props)
    ax.annotate('', xy=(0.50, 0.20), xytext=(0.50, 0.30), arrowprops=arrow_props)
    ax.annotate('', xy=(0.80, 0.20), xytext=(0.65, 0.30), arrowprops=arrow_props)

    plt.tight_layout()
    plt.savefig("figures/fig1_study_design.pdf", bbox_inches="tight")
    plt.savefig("figures/fig1_study_design.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved Figure 1 (PDF + PNG)")


def generate_figure2_regional_vulnerability():
    """Figure 2: regional progression scores vs the a priori Braak hierarchy (single split vs repeated CV)."""
    h1 = load_json("results/h1_regional.json"); hr = load_json("results/h1_robust.json")
    reg_data = h1.get("regional_results_msbb", {})
    if not reg_data or not hr:
        print("Warning: H1 results missing, skipping Figure 2"); return
    adj = hr["variants"]["adjusted"]; plain = hr["variants"]["plain"]
    df = pd.DataFrame.from_dict(reg_data, orient="index")
    df["rep_adj"] = [adj["regional_rho"][r] for r in df.index]
    df["lo"] = [adj["regional_ci"][r][0] for r in df.index]; df["hi"] = [adj["regional_ci"][r][1] for r in df.index]
    df = df.sort_values(["apriori_vulnerability_rank", "rep_adj"], ascending=[True, True])
    present = sorted(df["apriori_vulnerability_rank"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), dpi=300, gridspec_kw={'width_ratios': [1.3, 1]})
    ax1 = axes[0]
    pal = {1: '#c0392b', 2: '#e67e22', 3: '#2980b9', 4: '#27ae60', 5: '#7f8c8d'}
    y_pos = np.arange(len(df))
    xerr = np.vstack([df["rep_adj"] - df["lo"], df["hi"] - df["rep_adj"]])
    ax1.barh(y_pos, df["rep_adj"], color=[pal[r] for r in df["apriori_vulnerability_rank"]], height=0.7,
             xerr=xerr, error_kw=dict(ecolor="#222222", elinewidth=0.9, capsize=2))
    ax1.scatter(df["cv_spearman_rho"], y_pos, marker="D", s=26, facecolor="white", edgecolor="black", zorder=5, label="Single split (pre-registered)")
    ax1.set_yticks(y_pos); ax1.set_yticklabels(df["region"])
    ax1.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax1.set_xlabel("Braak progression score (Spearman rho)")
    ax1.set_title("A. Repeated 5-fold CV x10, age/sex/PMI-adjusted (bars, 95% donor-bootstrap CI)", fontsize=9.5, weight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4, axis="x")
    from matplotlib.patches import Patch
    names = {1: "Tier 1 (Hippocampus)", 2: "Tier 2 (Temporal / limbic)", 3: "Tier 3 (Association cortex)", 4: "Tier 4 (Primary sensory/motor)", 5: "Tier 5 (Subcortical)"}
    handles = [Patch(facecolor=pal[t], label=names[t]) for t in present]
    handles.append(plt.Line2D([], [], marker="D", color="black", markerfacecolor="white", linestyle="", label="Single split (pre-registered)"))
    ax1.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8, frameon=False)

    ax2 = axes[1]
    ax2.scatter(df["apriori_vulnerability_rank"], df["cv_spearman_rho"], s=45, marker="D", facecolor="white", edgecolor="#555555", label="Single split")
    ax2.scatter(df["apriori_vulnerability_rank"], df["rep_adj"], s=50, color="#1f3b73", alpha=0.85, label="Repeated CV, adjusted")
    xs = np.array([1, 5])
    for col, c in [("cv_spearman_rho", "#999999"), ("rep_adj", "#d9534f")]:
        k, b0 = np.polyfit(df["apriori_vulnerability_rank"], df[col], 1)
        ax2.plot(xs, k * xs + b0, color=c, linewidth=2)
    ax2.set_xticks([1, 2, 3, 4, 5])
    ax2.set_xlabel("Pre-registered tier (1 = earliest Braak involvement, 5 = latest / protected)")
    ax2.set_ylabel("Regional progression score (Spearman rho)")
    ax2.set_title("B. Tier vs regional score", fontsize=11, weight="bold")
    txt = (f"Single split: rho = {h1['observed_rho_prediction']:.2f}, p = {h1['permutation_p_value']:.3f}\n"
           f"Repeated CV: rho = {plain['rho_H1']:.2f}, p = {plain['perm_p']:.2f}\n"
           f"Repeated CV, adjusted: rho = {adj['rho_H1']:.2f}, p = {adj['perm_p']:.2f}\n"
           f"Bootstrap 95% CI [{adj['boot_ci'][0]:.2f}, {adj['boot_ci'][1]:.2f}]\n"
           "(rho = correlation of 6 - tier with score;\np: 5,000 tier-label shuffles over regions)")
    ax2.text(0.98, 0.98, txt, transform=ax2.transAxes, ha="right", va="top", fontsize=7.8,
             bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec="#1f3b73", lw=1.0))
    ax2.set_ylim(-0.35, 0.78)
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("figures/fig2_regional_vulnerability.pdf", bbox_inches="tight")
    plt.savefig("figures/fig2_regional_vulnerability.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved Figure 2 (PDF + PNG)")


def generate_figure3_resilience():
    """Figure 3: Cognitive Resilience at Matched Neuropathology."""
    h2 = load_json("results/h2_resilience.json")
    if not h2:
        print("Warning: h2_resilience.json empty, skipping Figure 3")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Panel A: Regional Classification AUCs in Cortical Regions for Resilient vs Demented
    reg_res = h2.get("regional_classification_results", {})
    if reg_res:
        df_reg = pd.DataFrame.from_dict(reg_res, orient="index")
        ax1 = axes[0]
        y_pos = np.arange(len(df_reg))
        errs = np.vstack([
            df_reg["auc"] - df_reg["ci_lower"],
            df_reg["ci_upper"] - df_reg["auc"]
        ])
        ax1.errorbar(df_reg["auc"], y_pos, xerr=errs, fmt='o', color='#1f3b73', ecolor='#2980b9', elinewidth=2, capsize=4, markersize=7)
        ax1.axvline(0.50, color='#d9534f', linestyle='--', linewidth=1.2, label='Chance Level (0.50)')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(df_reg["region"])
        ax1.set_xlabel("Cross-Validated AUC (Resilient vs Demented)")
        ax1.set_title("A. Cortical Resilience Discrimination (Braak >= IV)", fontsize=11, weight="bold")
        ax1.set_xlim(0.0, 1.02)
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.legend(loc="lower right", fontsize=8.5)

    # Panel B: Cross-Cohort Signature Transfer (AsymAD -> MSBB Resilient)
    xfer = h2.get("cross_cohort_transfer", {})
    ax2 = axes[1]
    auc_val = xfer.get("transfer_auc", 0.0)
    ci_low = xfer.get("ci_lower", 0.0)
    ci_high = xfer.get("ci_upper", 0.0)

    # Plot AUC bar with bootstrap CI
    ax2.bar([0], [auc_val], yerr=[[auc_val - ci_low], [ci_high - auc_val]], color="#008080", width=0.4, capsize=6, edgecolor="#2c3e50")
    ax2.axhline(0.50, color="#d9534f", linestyle="--", linewidth=1.2, label="Chance (0.50)")
    ax2.set_xticks([0])
    ax2.set_xticklabels(["Cross-Cohort Transfer\n(GSE118553 AsymAD -> MSBB Resilient)"])
    ax2.set_ylabel("ROC AUC")
    ax2.set_ylim(0.3, 1.0)
    ax2.set_title(f"B. Independent Cross-Cohort Resilience Transfer\nAUC = {auc_val:.3f} (95% CI: [{ci_low:.3f}, {ci_high:.3f}])", fontsize=11, weight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax2.legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    plt.savefig("figures/fig3_resilience.pdf", bbox_inches="tight")
    plt.savefig("figures/fig3_resilience.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved Figure 3 (PDF + PNG)")


def generate_figure4_replication_leakage():
    """Figure 4: External Lockbox Replication & Leakage Taxonomy Benchmark."""
    h3 = load_json("results/h3_replication.json")
    e1 = load_json("results/e1_leakage.json")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Panel A: Internal CV vs External Lockbox AUCs
    ax1 = axes[0]
    internal_auc = h3.get("internal_cv_auc", 0.0)
    internal_ci = h3.get("internal_cv_ci", [0.0, 0.0])
    lockbox = h3.get("lockbox_replication_gse132903", {})
    lockbox_auc = lockbox.get("external_auc", 0.0)
    lockbox_low = lockbox.get("ci_lower", 0.0)
    lockbox_high = lockbox.get("ci_upper", 0.0)

    msbb_x = h3.get("cross_platform_replication_msbb", {})
    msbb_auc = msbb_x.get("external_auc", 0.0)
    msbb_low = msbb_x.get("ci_lower", 0.0)
    msbb_high = msbb_x.get("ci_upper", 0.0)

    ag = load_json("results/h3_age_checks.json")
    x_labels = ["Internal CV\n(GSE118553 TC)", "Lockbox\n(GSE132903)", "Lockbox:\nage+sex only", "Lockbox:\nage/sex-resid.\nmodel", "Cross-platform\n(MSBB temporal)"]
    a_as = ag.get("age_sex_transfer_auc", {}); a_rs = ag.get("age_sex_residualised_model_auc", {})
    aucs = [internal_auc, lockbox_auc, a_as.get("auc", np.nan), a_rs.get("auc", np.nan), msbb_auc]
    lows = [internal_ci[0], lockbox_low, a_as.get("ci_lower", np.nan), a_rs.get("ci_lower", np.nan), msbb_low]
    highs = [internal_ci[1], lockbox_high, a_as.get("ci_upper", np.nan), a_rs.get("ci_upper", np.nan), msbb_high]
    errs = np.vstack([np.array(aucs) - np.array(lows), np.array(highs) - np.array(aucs)])

    bars1 = ax1.bar(np.arange(5), aucs, yerr=errs, color=["#1f3b73", "#27ae60", "#95a5a6", "#58d68d", "#2980b9"], capsize=5, width=0.6, edgecolor="#2c3e50")
    ax1.axhline(0.50, color="#d9534f", linestyle="--", linewidth=1.2, label="Chance (0.50)")
    ax1.set_xticks(np.arange(5))
    ax1.set_xticklabels(x_labels, fontsize=7.5)
    ax1.set_ylabel("ROC AUC")
    ax1.set_ylim(0.4, 1.0)
    ax1.set_title("A. External lockbox replication and age/sex controls", fontsize=11, weight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax1.legend(loc="lower right", fontsize=8.5)

    # Panel B: Leakage Taxonomy Benchmark (Optimism Bias)
    ax2 = axes[1]
    leak_labels = ["Zero-leakage\nreference", "Normalization\nLeakage", "Feature Selection\nLeakage", "Donor\nLeakage", "Compound\nLeakage"]
    leak_aucs = [
        e1.get("gold_standard_zero_leakage_auc", 0.0),
        e1.get("normalization_leakage_auc", 0.0),
        e1.get("feature_selection_leakage_auc", 0.0),
        e1.get("donor_leakage_auc", 0.0),
        e1.get("compound_leakage_auc", 0.0)
    ]
    colors_leak = ["#2e7d32", "#f0ad4e", "#e67e22", "#d9534f", "#96281b"]
    ax2.bar(np.arange(5), leak_aucs, color=colors_leak, width=0.6, edgecolor="#2c3e50")
    ax2.set_xticks(np.arange(5))
    ax2.set_xticklabels(leak_labels, rotation=15, ha="right", fontsize=8.5)
    ax2.set_ylabel("Cross-Validated ROC AUC")
    ax2.set_ylim(0.5, 1.05)
    ax2.set_title("B. Leakage Taxonomy: Empirical Optimism Bias", fontsize=11, weight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")

    # Annotate inflation on compound bar
    inf = e1.get("compound_inflation", 0.0)
    ax2.text(4, leak_aucs[4] + 0.02, f"+{inf:.3f}\nInflation", ha="center", fontsize=8, weight="bold", color="#96281b")

    plt.tight_layout()
    plt.savefig("figures/fig4_replication_leakage.pdf", bbox_inches="tight")
    plt.savefig("figures/fig4_replication_leakage.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved Figure 4 (PDF + PNG)")


def generate_figure5_biology_celltype():
    """Figure 5: Cell-type marker scores (donor level) and overlap of Braak-associated genes with curated gene panels."""
    e2 = load_json("results/e2_celltype.json")
    e5 = load_json("results/e5_biology.json")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), dpi=300)

    # Panel A: donor-level Spearman correlation of each marker-panel score with Braak, with 95% donor-bootstrap CI
    ax1 = axes[0]
    cell_corrs = e2.get("donor_level_vs_braak", {})
    if cell_corrs:
        df_cell = pd.DataFrame.from_dict(cell_corrs, orient="index").sort_values("spearman_rho")
        y_pos = np.arange(len(df_cell))
        colors_ct = ['#d9534f' if r < 0 else '#2e7d32' for r in df_cell["spearman_rho"]]
        xerr = np.vstack([df_cell["spearman_rho"] - df_cell["ci_lower"], df_cell["ci_upper"] - df_cell["spearman_rho"]])
        ax1.barh(y_pos, df_cell["spearman_rho"], color=colors_ct, height=0.6, edgecolor="#2c3e50", xerr=xerr, error_kw=dict(ecolor="#222222", capsize=3))
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([s_.capitalize() for s_ in df_cell.index])
        ax1.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax1.set_xlabel("Spearman rho vs Braak stage (one point per donor, N = %d)" % e2.get("n_donors", 0))
        ax1.set_title("A. Marker-panel scores vs Braak, donor level", fontsize=11, weight="bold")
        ax1.grid(True, linestyle="--", alpha=0.4, axis="x")

    # Panel B: overlap of the top/bottom-300 Braak-correlated genes with curated gene panels (log-scale odds ratio)
    ax2 = axes[1]
    up_mod = e5.get("upregulated_module_enrichment", {}); dn_mod = e5.get("downregulated_module_enrichment", {})
    if up_mod and dn_mod:
        names = list(up_mod.keys()); y_pos2 = np.arange(len(names)); h = 0.36
        floor = 0.3
        up_or = [max(up_mod[k]["odds_ratio"], floor) for k in names]; dn_or = [max(dn_mod[k]["odds_ratio"], floor) for k in names]
        ax2.barh(y_pos2 + h / 2, up_or, height=h, color="#d35400", edgecolor="#2c3e50", label="Genes rising with Braak")
        ax2.barh(y_pos2 - h / 2, dn_or, height=h, color="#1f3b73", edgecolor="#2c3e50", label="Genes falling with Braak")
        ax2.set_xscale("log"); ax2.set_xlim(0.25, 90)
        ax2.set_yticks(y_pos2); ax2.set_yticklabels([k.replace("_", " ") for k in names])
        for i, k in enumerate(names):
            for off, m in [(h / 2, up_mod[k]), (-h / 2, dn_mod[k])]:
                lab = f"{m['overlap_count']}/{m['module_size_in_bg']}" + ("*" if m["bonferroni_p"] < 0.05 else "")
                ax2.text(max(m["odds_ratio"], floor) * 1.08, i + off, lab, va="center", fontsize=7.5)
        ax2.axvline(1.0, color="#d9534f", linestyle="--", linewidth=1.0)
        ax2.set_xlabel("Fisher odds ratio (log scale; bars at left edge = no overlap)")
        ax2.set_title("B. Braak-associated genes vs curated gene panels\n(labels = overlap / panel size; * Bonferroni p < 0.05)", fontsize=10, weight="bold")
        ax2.grid(True, linestyle="--", alpha=0.4, axis="x")
        ax2.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig("figures/fig5_biology_celltype.pdf", bbox_inches="tight")
    plt.savefig("figures/fig5_biology_celltype.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved Figure 5 (PDF + PNG)")


def main():
    print("=== GENERATING ALL 5 PUBLICATION-GRADE FIGURES ===")
    os.makedirs("figures", exist_ok=True)
    generate_figure1_study_design()
    generate_figure2_regional_vulnerability()
    generate_figure3_resilience()
    generate_figure4_replication_leakage()
    generate_figure5_biology_celltype()
    print("All figures successfully generated in figures/ directory!")


if __name__ == "__main__":
    main()
