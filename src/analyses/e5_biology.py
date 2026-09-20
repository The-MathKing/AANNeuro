"""
Exploratory Analysis E5: Biological Anchoring & Consensus Module Overlap.
Anchors discovery and resilient transcriptomic signatures to:
1. Validated AD GWAS Risk Genes (Bellenguez et al. Nature Genetics 2022, Lambert et al. 2013)
2. AMP-AD 30 Consensus Co-Expression Modules (Wan et al. Cell Systems 2020: Immune/Microglial, Neuronal/Synaptic, ECM/Endothelial, Mitochondrial)
3. Curated KEGG/GO Hallmark Neurodegenerative Pathways (Synaptic vesicle cycle, Neuroinflammation, Oxidative Phosphorylation, Axon Guidance)
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats

from src.load import load_gse118553, load_gse84422
from src.eval.ledger_logger import log_analysis

# Validated AD GWAS Risk Loci from Bellenguez et al. (2022) & Lambert et al. (2013)
AD_GWAS_GENES = [
    "APOE", "BIN1", "CLU", "ABCA7", "CR1", "PICALM", "MS4A6A", "CD33", "MS4A4A", "CD2AP",
    "EPHA1", "PTK2B", "CASS4", "INPP5D", "MEF2C", "NME8", "ZCWPW1", "CERCAM", "FERMT2", "SLC24A4",
    "SORL1", "ADAM10", "TREM2", "PLCG2", "ABI3", "APH1B", "APP", "PSEN1", "PSEN2", "MAPT",
    "ADAMTS1", "USP6NL", "IL34", "SHARPIN", "ICA1L", "GRN", "TREML2", "UNC5C", "SCIMP", "BLNK"
]

# AMP-AD Consensus Biological Modules (Wan et al. 2020)
AMP_AD_MODULES = {
    "Immune_Microglial": [
        "TYROBP", "TREM2", "AIF1", "CD68", "C1QA", "C1QB", "C1QC", "FCER1G", "ITGAM", "MS4A6A", "HLA-DRA", "CX3CR1", "SPI1", "TLR2", "LAPTM5"
    ],
    "Neuronal_Synaptic": [
        "SYN1", "SYN2", "SYP", "SNAP25", "VAMP2", "STX1A", "BSN", "PCLO", "SLC17A7", "CAMK2A", "GRIN1", "GRIN2A", "GRIA1", "DLG4", "NEFL"
    ],
    "ECM_Endothelial_Vascular": [
        "COL1A1", "COL1A2", "COL3A1", "COL4A1", "COL4A2", "FN1", "VWF", "PECAM1", "CLDN5", "CD34", "CDH5", "LAMB1", "LAMA4", "SPARC", "BGN"
    ],
    "Mitochondrial_Metabolic": [
        "NDUFA1", "NDUFB1", "NDUFS1", "SDHA", "SDHB", "UQCRC1", "UQCRC2", "COX4I1", "COX5A", "ATP5F1A", "ATP5F1B", "SOD2", "PRDX3", "VDAC1"
    ]
}


def compute_pathway_enrichment(query_genes, background_genes, gene_sets):
    """
    Compute Fisher's Exact Test / Hypergeometric overlap enrichment.
    """
    q_set = set(query_genes)
    bg_set = set(background_genes)
    q_in_bg = q_set.intersection(bg_set)
    
    n_query = len(q_in_bg)
    n_bg = len(bg_set)
    
    results = {}
    for set_name, set_genes in gene_sets.items():
        s_in_bg = set(set_genes).intersection(bg_set)
        n_set = len(s_in_bg)
        
        overlap = q_in_bg.intersection(s_in_bg)
        k_overlap = len(overlap)
        
        # 2x2 contingency table:
        # [[in_query & in_set,     in_query & not_in_set],
        #  [not_in_query & in_set, not_in_query & not_in_set]]
        table = [
            [k_overlap, n_query - k_overlap],
            [n_set - k_overlap, n_bg - n_query - (n_set - k_overlap)]
        ]
        
        try:
            odds_ratio, p_value = stats.fisher_exact(table, alternative="greater")
        except Exception:
            odds_ratio, p_value = 1.0, 1.0
            
        results[set_name] = {
            "module_size_in_bg": n_set,
            "overlap_count": k_overlap,
            "overlap_genes": sorted(list(overlap)),
            "odds_ratio": float(odds_ratio) if not np.isinf(odds_ratio) else 999.0,
            "p_value": float(p_value)
        }
        
    return results


def run_e5_analysis():
    print("=== RUNNING EXPLORATORY ANALYSIS E5: BIOLOGICAL ANCHORING ===")
    os.makedirs("results", exist_ok=True)

    meta_msbb, expr_msbb = load_gse84422()
    # Find top disease-associated genes in MSBB
    valid_msbb = meta_msbb.dropna(subset=["braak"]).copy()
    y_braak = valid_msbb["braak"].values
    
    # Compute gene correlations with Braak
    rhos = {}
    for g in expr_msbb.index:
        vals = expr_msbb.loc[g, valid_msbb.index].values
        r, _ = stats.spearmanr(y_braak, vals)
        if not np.isnan(r):
            rhos[g] = r
            
    df_rhos = pd.Series(rhos).sort_values(ascending=False)
    upregulated_ad_genes = df_rhos.head(300).index.tolist()
    downregulated_ad_genes = df_rhos.tail(300).index.tolist()
    background_genes = expr_msbb.index.tolist()

    # 1. Overlap with AMP-AD Consensus Modules
    enrich_up = compute_pathway_enrichment(upregulated_ad_genes, background_genes, AMP_AD_MODULES)
    enrich_down = compute_pathway_enrichment(downregulated_ad_genes, background_genes, AMP_AD_MODULES)

    # 2. Overlap with AD GWAS Risk Loci
    gwas_dict = {"AD_GWAS_Risk_Loci": AD_GWAS_GENES}
    gwas_enrich = compute_pathway_enrichment(upregulated_ad_genes + downregulated_ad_genes, background_genes, gwas_dict)

    print("\nBiological Module Enrichments for Upregulated Disease Signature:")
    for mod, res in enrich_up.items():
        print(f"  {mod:25s}: Overlap={res['overlap_count']}/{res['module_size_in_bg']}, OR={res['odds_ratio']:.2f}, p={res['p_value']:.2e} ({res['overlap_genes'][:4]})")

    print("\nBiological Module Enrichments for Downregulated Disease Signature:")
    for mod, res in enrich_down.items():
        print(f"  {mod:25s}: Overlap={res['overlap_count']}/{res['module_size_in_bg']}, OR={res['odds_ratio']:.2f}, p={res['p_value']:.2e} ({res['overlap_genes'][:4]})")

    print("\nAD GWAS Risk Loci Enrichment:")
    gwas_res = gwas_enrich["AD_GWAS_Risk_Loci"]
    print(f"  GWAS Overlap={gwas_res['overlap_count']}/{gwas_res['module_size_in_bg']}, OR={gwas_res['odds_ratio']:.2f}, p={gwas_res['p_value']:.2e} ({gwas_res['overlap_genes']})")

    # Bonferroni across all 9 Fisher tests reported (4 modules x up/down + GWAS)
    all_p = [r["p_value"] for r in enrich_up.values()] + [r["p_value"] for r in enrich_down.values()] + [gwas_res["p_value"]]
    n_tests = len(all_p)
    for r in list(enrich_up.values()) + list(enrich_down.values()) + [gwas_res]:
        r["bonferroni_p"] = float(min(1.0, r["p_value"] * n_tests))

    # Sensitivity: rank genes at the donor level (region-centred profile per donor; Spearman vs Braak across donors)
    Xc = expr_msbb[valid_msbb.index].T.copy()
    for reg, idx in valid_msbb.groupby("region").groups.items():
        sub_ = Xc.loc[idx]
        Xc.loc[idx] = (sub_ - sub_.mean()) / sub_.std(ddof=0).replace(0, 1.0)
    Xc["donor_id"] = valid_msbb["donor_id"].values
    Xd = Xc.groupby("donor_id").mean()
    bd = valid_msbb.groupby("donor_id")["braak"].first().loc[Xd.index]
    rk = Xd.rank(axis=0).values; rb = stats.rankdata(bd.values)
    rk_c = rk - rk.mean(0); rb_c = rb - rb.mean()
    rho_d = pd.Series((rk_c * rb_c[:, None]).sum(0) / np.sqrt((rk_c ** 2).sum(0) * (rb_c ** 2).sum()), index=Xd.columns).dropna().sort_values(ascending=False)
    up_d, down_d = rho_d.head(300).index.tolist(), rho_d.tail(300).index.tolist()
    enrich_up_d = compute_pathway_enrichment(up_d, background_genes, AMP_AD_MODULES)
    enrich_down_d = compute_pathway_enrichment(down_d, background_genes, AMP_AD_MODULES)
    print("Donor-level-ranking sensitivity: synaptic (down) OR=%.1f p=%.2e; ECM (up) OR=%.1f p=%.2e" % (
        enrich_down_d["Neuronal_Synaptic"]["odds_ratio"], enrich_down_d["Neuronal_Synaptic"]["p_value"],
        enrich_up_d["ECM_Endothelial_Vascular"]["odds_ratio"], enrich_up_d["ECM_Endothelial_Vascular"]["p_value"]))

    results = {
        "analysis": "E5_Biological_Anchoring",
        "gene_list_definition": {
            "contrast": "continuous Braak stage (0-VI); Spearman correlation of each gene with Braak across all MSBB samples with valid Braak (pooled over 19 regions, no covariate adjustment)",
            "n_samples_ranked": int(len(valid_msbb)),
            "disease_up": "300 genes with the largest positive Spearman rho",
            "disease_down": "300 genes with the most negative Spearman rho",
            "threshold": "fixed list length (top/bottom 300 = 2.3% of the background each); no p-value threshold",
            "background": f"all {len(background_genes)} genes measured in the harmonised MSBB matrix",
            "test": "one-sided Fisher exact test, unadjusted p and Bonferroni over %d tests" % n_tests,
            "gene_panels": "hand-curated 11-15-gene lists per functional category, labelled after the AMP-AD consensus modules; they are NOT the modules' memberships",
        },
        "n_tests_bonferroni": n_tests,
        "sensitivity_donor_level_ranking": {"upregulated_module_enrichment": enrich_up_d, "downregulated_module_enrichment": enrich_down_d},
        "n_background_genes": len(background_genes),
        "upregulated_module_enrichment": enrich_up,
        "downregulated_module_enrichment": enrich_down,
        "gwas_risk_loci_enrichment": gwas_enrich
    }

    with open("results/e5_biology.json", "w") as f:
        json.dump(results, f, indent=2)

    log_analysis(
        analysis_id="E5_BIO_001",
        category="EXPLORATORY",
        hypothesis_or_idea="E5 Biological Anchoring (AMP-AD & GWAS Overlap)",
        description="Fisher's exact test overlap of transcriptomic signatures with AMP-AD consensus modules and AD GWAS loci",
        cohorts_used="GSE84422_MSBB",
        regions_analyzed="Genome-wide",
        model_type="Fisher_Exact_Enrichment",
        covariates_adjusted="None",
        sample_size_donors=valid_msbb["donor_id"].nunique(),
        primary_metric="Immune_Module_Odds_Ratio",
        point_estimate=enrich_up["Immune_Microglial"]["odds_ratio"],
        ci_95_lower=float('nan'),
        ci_95_upper=float('nan'),
        p_value=enrich_up["Immune_Microglial"]["p_value"],
        replicated_in_lockbox=False,
        status="COMPLETE",
        notes=f"Immune/Microglial (up) OR={enrich_up['Immune_Microglial']['odds_ratio']:.1f}, p={enrich_up['Immune_Microglial']['p_value']:.2e} (not significant); Synaptic (down) OR={enrich_down['Neuronal_Synaptic']['odds_ratio']:.1f}, p={enrich_down['Neuronal_Synaptic']['p_value']:.2e}. Gene panels are hand-curated, not AMP-AD module memberships."
    )

    return results


if __name__ == "__main__":
    run_e5_analysis()
