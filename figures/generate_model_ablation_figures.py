"""
Figure 6 (post hoc): sensitivity of lockbox and cross-platform AUC to model specification.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve, auc

sys.path.insert(0, os.path.abspath("."))

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
})

def generate_fig5():
    """Figure 6 (post hoc): lockbox AUC across eight model specifications, and
    cross-platform transfer across signature sizes. Both panels reuse cohorts that
    had already been evaluated, so they show sensitivity to modelling choices, not
    new independent validations."""
    print("Generating Figure 6: post hoc model-specification sensitivity...")

    with open("results/model_ablation_results.json", "r") as f:
        data = json.load(f)

    summary = data["benchmark_summary"]
    singscore_sweep = data["singscore_sweep"]

    short = {
        "baseline_h3": "TC, 1,000 HVG (C = 1)",
        "biological_priors_only": "TC, curated panels only (80 genes)",
        "hybrid_priors": "TC, 500 HVG + curated panels",
        "rank_transform_only": "TC, 1,000 HVG, within-sample ranks",
        "multi_region_pooled": "TC+EC+FC pooled, 1,000 HVG",
        "composite_a_pooled_hybrid": "Pooled + curated panels",
        "composite_b_pooled_rank": "Pooled + ranks",
        "composite_c_full_synergy": "Pooled + curated panels + ranks",
    }

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"width_ratios": [1.25, 1]})
    plt.subplots_adjust(wspace=0.55)

    # Panel A: forest plot of lockbox AUCs, in a fixed (not performance-sorted) order
    order = list(short.keys())
    rows = {r["id"]: r for r in summary}
    y_pos = np.arange(len(order))[::-1]
    for y_i, m_id in zip(y_pos, order):
        r = rows[m_id]
        ax_a.errorbar(r["lockbox_auc"], y_i,
                      xerr=[[r["lockbox_auc"] - r["lockbox_ci_lower"]], [r["lockbox_ci_upper"] - r["lockbox_auc"]]],
                      fmt="o", color="#1f3b6f", ecolor="#1f3b6f", elinewidth=1.6, capsize=3, markersize=5)
    lo = min(r["lockbox_ci_lower"] for r in summary)
    hi = max(r["lockbox_ci_upper"] for r in summary)
    ax_a.axvspan(min(r["lockbox_auc"] for r in summary), max(r["lockbox_auc"] for r in summary),
                 color="#999999", alpha=0.15, lw=0, label="Range of point estimates")
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels([short[m] for m in order])
    ax_a.set_xlim([0.70, 1.0])
    ax_a.set_xlabel("Lockbox AUC, GSE132903 (95% bootstrap CI)")
    ax_a.set_title("A  Lockbox AUC across model specifications", fontweight="bold", loc="left")
    ax_a.grid(True, linestyle="--", axis="x")
    ax_a.legend(loc="upper right", frameon=True, framealpha=0.9, facecolor="white")

    # Panel B: cross-platform transfer across signature sizes (no "optimum" is marked:
    # K was varied on the evaluation cohort itself)
    df_sing = pd.DataFrame(singscore_sweep)
    k_vals = df_sing["k"].values
    ax_b.errorbar(k_vals, df_sing["auc"].values,
                  yerr=[df_sing["auc"].values - df_sing["ci_lower"].values, df_sing["ci_upper"].values - df_sing["auc"].values],
                  fmt="o-", color="#8c564b", ecolor="#8c564b", elinewidth=1.2, capsize=3, markersize=5, linewidth=1.4)
    ax_b.axhline(0.50, color="#d62728", linestyle="--", linewidth=1.0, label="Chance (0.50)")
    ax_b.set_xscale("log")
    ax_b.set_xticks(k_vals)
    ax_b.set_xticklabels([str(k) for k in k_vals])
    ax_b.minorticks_off()
    ax_b.set_ylim([0.40, 0.80])
    ax_b.set_xlabel("Signature size K (top and bottom K genes)")
    ax_b.set_ylabel("AUC, MSBB temporal (Braak ≥ IV)")
    ax_b.set_title("B  Cross-platform transfer vs signature size", fontweight="bold", loc="left")
    ax_b.grid(True, linestyle="--")
    ax_b.legend(loc="lower right", frameon=True, framealpha=0.9, facecolor="white")

    out_pdf = "figures/fig6_model_ablation.pdf"
    out_png = "figures/fig6_model_ablation.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved Figure 6 to {out_pdf} and {out_png}")


if __name__ == "__main__":
    generate_fig5()
