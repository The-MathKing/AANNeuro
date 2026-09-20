"""
GSE84422 Mount Sinai Brain Bank (MSBB) Cohort Loader (Wang et al. 2016).
Multi-region microarray dataset covering 19 distinct post-mortem human brain regions:
- 17 regions profiled on Affymetrix HG-U133A (GPL96) and HG-U133B (GPL97)
- 2 regions (BM36 parahippocampal gyrus, BM44 inferior frontal gyrus) on HG-U133_Plus_2 (GPL570)
Includes donor-level Clinical Dementia Rating (CDR), Braak score (0-VI), CERAD rating, plaque/tangle counts, age, sex, PMI, and pH.
"""

import os
import re
import pandas as pd
import numpy as np
from src.load.geo_parser import parse_series_matrix, parse_platform_annotation


def _standardize_region_name(raw_name):
    """Map raw MSBB region descriptions to canonical Brodmann / anatomical identifiers."""
    name = str(raw_name).strip()
    mapping = {
        "Frontal Pole": "BM10_FP",
        "Superior Temporal Gyrus": "BM22_STG",
        "Inferior Temporal Gyrus": "BM20_ITG",
        "Parahippocampal Gyrus": "BM36_PHG",
        "Inferior Frontal Gyrus": "BM44_IFG",
        "Dorsolateral Prefrontal Cortex": "BM9_DLPFC",
        "Superior Parietal Lobule": "BM7_SPL",
        "Inferior Parietal Lobule": "BM39_IPL",
        "Anterior Cingulate Cortex": "BM24_ACC",
        "Posterior Cingulate Cortex": "BM23_PCC",
        "Primary Visual Cortex": "BM17_PVC",
        "Primary Motor Cortex": "BM4_PMC",
        "Primary Auditory Cortex": "BM41_42_PAC",
        "Temporal Pole": "BM38_TP",
        "Entorhinal Cortex": "BM28_EC",
        "Hippocampus": "HIP",
        "Amygdala": "AMY",
        "Caudate Nucleus": "CAUD",
        "Putamen": "PUT",
        "Cerebellum": "CRB"
    }
    for k, v in mapping.items():
        if k.lower() in name.lower():
            return v
    return name.replace(" ", "_")


def _clean_braak(raw):
    """Braak NFT stage is defined on 0-VI. Values outside that range (e.g. the code 9 found for
    donor 1009 in GSE84422) are missing-data sentinels, not stages, and are set to NaN."""
    b = pd.to_numeric(raw, errors="coerce")
    return b.where((b >= 0) & (b <= 6))


def load_gse84422(raw_dir="data/raw", include_all_platforms=True):
    """
    Load and harmonize GSE84422 MSBB dataset.
    Returns:
        meta_df: DataFrame of sample metadata indexed by GSM ID.
        expr_df: DataFrame of gene expression (genes x samples).
    """
    gpl96_mat = os.path.join(raw_dir, "GSE84422-GPL96_series_matrix.txt.gz")
    gpl96_ann = os.path.join(raw_dir, "GPL96_table.txt.gz")
    gpl570_mat = os.path.join(raw_dir, "GSE84422-GPL570_series_matrix.txt.gz")
    gpl570_ann = os.path.join(raw_dir, "GPL570_table.txt.gz")

    if not os.path.exists(gpl96_mat):
        raise FileNotFoundError(f"GSE84422-GPL96 missing at {gpl96_mat}")

    # 1. Load GPL96
    meta_96_raw, expr_96_raw = parse_series_matrix(gpl96_mat, load_expression=True)
    annot_96 = parse_platform_annotation(gpl96_ann)
    p2g_96 = annot_96.set_index("probe_id")["gene_symbol"].to_dict()

    expr_96_raw["gene_symbol"] = expr_96_raw.index.map(p2g_96)
    expr_96_valid = expr_96_raw.dropna(subset=["gene_symbol"])
    expr_96_valid = expr_96_valid[expr_96_valid["gene_symbol"] != ""]
    expr_96_gene = expr_96_valid.groupby("gene_symbol").mean()

    # Clean metadata for GPL96
    meta_96 = pd.DataFrame(index=meta_96_raw.index)
    meta_96["sample_id"] = meta_96_raw.index
    meta_96["cohort"] = "GSE84422_MSBB"
    meta_96["platform"] = "GPL96"
    meta_96["donor_id"] = meta_96_raw["subject id"].astype(str).str.strip()
    meta_96["raw_region"] = meta_96_raw["brain region"].astype(str).str.strip()
    meta_96["region"] = meta_96["raw_region"].map(_standardize_region_name)
    meta_96["age"] = pd.to_numeric(meta_96_raw["age"].astype(str).str.replace("+", "", regex=False), errors="coerce")
    meta_96["sex"] = meta_96_raw["Sex"].astype(str).str.upper().map(lambda s: "FEMALE" if "FEMALE" in s or s == "F" else "MALE")
    meta_96["pmi"] = pd.to_numeric(meta_96_raw["postmortem interval minutes"], errors="coerce")
    meta_96["ph"] = pd.to_numeric(meta_96_raw["ph"], errors="coerce")
    meta_96["cdr"] = pd.to_numeric(meta_96_raw["clinical dementia rating"], errors="coerce")
    meta_96["braak"] = _clean_braak(meta_96_raw["braak neurofibrillary tangle score"])
    meta_96["cerad"] = pd.to_numeric(meta_96_raw["sum of cerad rating scores in multiple brain regions"], errors="coerce")
    meta_96["plaque_density"] = pd.to_numeric(meta_96_raw.get("average neuritic plaque density", np.nan), errors="coerce")
    meta_96["tangle_density"] = pd.to_numeric(meta_96_raw.get("sum of neurofibrillary tangles density in multiple brain regions", np.nan), errors="coerce")
    meta_96["neuropath_category"] = meta_96_raw.get("neuropathological category", "Unknown")

    if not include_all_platforms or not os.path.exists(gpl570_mat):
        common = meta_96.index.intersection(expr_96_gene.columns)
        return meta_96.loc[common], expr_96_gene[common]

    # 2. Load GPL570
    meta_570_raw, expr_570_raw = parse_series_matrix(gpl570_mat, load_expression=True)
    annot_570 = parse_platform_annotation(gpl570_ann)
    p2g_570 = annot_570.set_index("probe_id")["gene_symbol"].to_dict()

    expr_570_raw["gene_symbol"] = expr_570_raw.index.map(p2g_570)
    expr_570_valid = expr_570_raw.dropna(subset=["gene_symbol"])
    expr_570_valid = expr_570_valid[expr_570_valid["gene_symbol"] != ""]
    expr_570_gene = expr_570_valid.groupby("gene_symbol").mean()

    meta_570 = pd.DataFrame(index=meta_570_raw.index)
    meta_570["sample_id"] = meta_570_raw.index
    meta_570["cohort"] = "GSE84422_MSBB"
    meta_570["platform"] = "GPL570"
    meta_570["donor_id"] = meta_570_raw["subject id"].astype(str).str.strip()
    meta_570["raw_region"] = meta_570_raw["brain region"].astype(str).str.strip()
    meta_570["region"] = meta_570["raw_region"].map(_standardize_region_name)
    meta_570["age"] = pd.to_numeric(meta_570_raw["age"].astype(str).str.replace("+", "", regex=False), errors="coerce")
    meta_570["sex"] = meta_570_raw["Sex"].astype(str).str.upper().map(lambda s: "FEMALE" if "FEMALE" in s or s == "F" else "MALE")
    meta_570["pmi"] = pd.to_numeric(meta_570_raw["postmortem interval minutes"], errors="coerce")
    meta_570["ph"] = pd.to_numeric(meta_570_raw["ph"], errors="coerce")
    meta_570["cdr"] = pd.to_numeric(meta_570_raw["clinical dementia rating"], errors="coerce")
    meta_570["braak"] = _clean_braak(meta_570_raw["braak neurofibrillary tangle score"])
    meta_570["cerad"] = pd.to_numeric(meta_570_raw["sum of cerad rating scores in multiple brain regions"], errors="coerce")
    meta_570["plaque_density"] = pd.to_numeric(meta_570_raw.get("average neuritic plaque density", np.nan), errors="coerce")
    meta_570["tangle_density"] = pd.to_numeric(meta_570_raw.get("sum of neurofibrillary tangles density in multiple brain regions", np.nan), errors="coerce")
    meta_570["neuropath_category"] = meta_570_raw.get("neuropathological category", "Unknown")

    # Combine metadata
    meta_combined = pd.concat([meta_96, meta_570], axis=0)

    # Intersect genes across platforms
    common_genes = expr_96_gene.index.intersection(expr_570_gene.index)
    expr_combined = pd.concat([expr_96_gene.loc[common_genes], expr_570_gene.loc[common_genes]], axis=1)

    common_samples = meta_combined.index.intersection(expr_combined.columns)
    meta_combined = meta_combined.loc[common_samples]
    expr_combined = expr_combined[common_samples]

    return meta_combined, expr_combined
