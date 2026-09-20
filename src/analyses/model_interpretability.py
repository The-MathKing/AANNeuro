"""
Biological Interpretability Engine: Decomposing predictive model weights into functional biological pathways.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from src.analyses.e5_biology import AMP_AD_MODULES, AD_GWAS_GENES

def run_interpretability_analysis():
    print("\n" + "=" * 75)
    print("RUNNING BIOLOGICAL MODEL INTERPRETABILITY ANALYSIS")
    print("=" * 75)

    results_path = "results/model_ablation_results.json"
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Ablation results missing at {results_path}")

    with open(results_path, "r") as f:
        ablation_data = json.load(f)

    model_weights = ablation_data.get("model_weights", {})

    # Categorize genes into functional biological modules
    categories = {
        "Synaptic / Neuronal": AMP_AD_MODULES.get("synaptic", []),
        "Microglial / Immune": AMP_AD_MODULES.get("microglial_immune", []),
        "Astrocytic / ECM": AMP_AD_MODULES.get("astrocytic_inflammatory", []),
        "Mitochondrial / Metabolic": AMP_AD_MODULES.get("myelin_metabolic", []),
        "AD GWAS Loci": AD_GWAS_GENES
    }

    interpretability_summary = {}

    for model_id, weights_dict in model_weights.items():
        s = pd.Series(weights_dict)
        if len(s) == 0:
            continue

        # Top positive (up in AD) and top negative (down in AD / preserved in CTL)
        top_pos = s.nlargest(15).to_dict()
        top_neg = s.nsmallest(15).to_dict()

        # Aggregate magnitude by biological category
        cat_stats = {}
        for cat_name, cat_genes in categories.items():
            avail = [g for g in cat_genes if g in s.index]
            if len(avail) > 0:
                sub_weights = s.loc[avail]
                cat_stats[cat_name] = {
                    "n_genes_present": len(avail),
                    "mean_weight": float(sub_weights.mean()),
                    "sum_weight": float(sub_weights.sum()),
                    "mean_abs_weight": float(np.abs(sub_weights).mean()),
                    "top_gene": sub_weights.abs().idxmax(),
                    "top_gene_weight": float(sub_weights.loc[sub_weights.abs().idxmax()])
                }

        interpretability_summary[model_id] = {
            "top_positive_weights_ad": top_pos,
            "top_negative_weights_ctl": top_neg,
            "category_breakdown": cat_stats
        }

        print(f"\nModel: {model_id}")
        print("  Top AD-elevated genes:", list(top_pos.keys())[:5])
        print("  Top CTL-preserved/depleted genes:", list(top_neg.keys())[:5])
        for cat_name, cstat in cat_stats.items():
            print(f"  [{cat_name:25s}] (N={cstat['n_genes_present']:2d}) Mean Beta = {cstat['mean_weight']:+.3f}, Top: {cstat['top_gene']} ({cstat['top_gene_weight']:+.3f})")

    out_file = "results/model_interpretability.json"
    with open(out_file, "w") as f:
        json.dump(interpretability_summary, f, indent=2)

    print(f"\nSaved interpretability breakdown to {out_file}")


if __name__ == "__main__":
    run_interpretability_analysis()
