"""
GSE118553 Cohort Loader (Patel et al. 2019).
Illumina HumanHT-12 V4.0 array across 4 brain regions:
Temporal Cortex, Frontal Cortex, Entorhinal Cortex, Cerebellum.
"""

import os
import re
import pandas as pd
import numpy as np
from src.load.geo_parser import parse_series_matrix, parse_platform_annotation


def load_gse118553(raw_dir="data/raw", min_expression=0.0):
    """
    Load and harmonize GSE118553 dataset.
    Returns:
        meta_df: DataFrame with sample metadata indexed by sample accession.
        expr_df: DataFrame with gene expression (genes x samples).
    """
    matrix_path = os.path.join(raw_dir, "GSE118553_series_matrix.txt.gz")
    annot_path = os.path.join(raw_dir, "GPL10558.annot.gz")

    if not os.path.exists(matrix_path):
        raise FileNotFoundError(f"GSE118553 series matrix missing at {matrix_path}")
    if not os.path.exists(annot_path):
        raise FileNotFoundError(f"GPL10558 annotation missing at {annot_path}")

    meta_raw, expr_raw = parse_series_matrix(matrix_path, load_expression=True)
    annot = parse_platform_annotation(annot_path)

    # Standardize metadata
    meta = pd.DataFrame(index=meta_raw.index)
    meta["sample_id"] = meta_raw.index
    meta["cohort"] = "GSE118553"
    meta["platform"] = "GPL10558"

    # Donor ID extraction from "individual: Subject  X" or title
    if "individual" in meta_raw.columns:
        meta["donor_id"] = meta_raw["individual"].astype(str).str.strip()
    elif "individual_id" in meta_raw.columns:
        meta["donor_id"] = meta_raw["individual_id"].astype(str).str.strip()
    else:
        # Fallback extract from title: e.g. "Temporal_Cortex AD (9371242065_I)"
        meta["donor_id"] = meta_raw["title"].astype(str)

    # Clean donor IDs: normalize whitespace
    meta["donor_id"] = meta["donor_id"].apply(lambda s: re.sub(r"\s+", " ", str(s)).strip())

    # Tissue / Region
    if "tissue" in meta_raw.columns:
        meta["region"] = meta_raw["tissue"].astype(str).str.strip()
    else:
        meta["region"] = meta_raw["title"].apply(lambda t: str(t).split()[0] if pd.notna(t) else "Unknown")

    # Standardize region names
    region_map = {
        "Temporal_Cortex": "Temporal_Cortex",
        "Frontal_Cortex": "Frontal_Cortex",
        "Entorhinal_Cortex": "Entorhinal_Cortex",
        "Cerebellum": "Cerebellum",
        "temporal cortex": "Temporal_Cortex",
        "frontal cortex": "Frontal_Cortex",
        "entorhinal cortex": "Entorhinal_Cortex",
        "cerebellum": "Cerebellum"
    }
    meta["region"] = meta["region"].map(lambda r: region_map.get(r, r))

    # Disease status
    if "disease state" in meta_raw.columns:
        diag_raw = meta_raw["disease state"].astype(str).str.strip()
    elif "diagnosis" in meta_raw.columns:
        diag_raw = meta_raw["diagnosis"].astype(str).str.strip()
    else:
        diag_raw = pd.Series("Unknown", index=meta_raw.index)

    meta["diagnosis"] = diag_raw.map(
        lambda d: "CTL" if str(d).lower() in ["control", "ctl", "normal", "nd"] 
        else ("AsymAD" if "asym" in str(d).lower() 
        else ("AD" if "ad" in str(d).lower() else str(d)))
    )

    # Age & Sex
    if "age" in meta_raw.columns:
        meta["age"] = pd.to_numeric(meta_raw["age"].astype(str).str.replace("+", "", regex=False), errors="coerce")
    else:
        meta["age"] = np.nan

    if "gender" in meta_raw.columns:
        meta["sex"] = meta_raw["gender"].astype(str).str.upper().str.strip()
    elif "sex" in meta_raw.columns:
        meta["sex"] = meta_raw["sex"].astype(str).str.upper().str.strip()
    else:
        meta["sex"] = "UNKNOWN"
    meta["sex"] = meta["sex"].map(lambda s: "FEMALE" if "FEMALE" in str(s) or s == "F" else ("MALE" if "MALE" in str(s) or s == "M" else "UNKNOWN"))

    # Map probes to gene symbols and aggregate by mean
    probe_to_gene = annot.set_index("probe_id")["gene_symbol"].to_dict()
    expr_raw["gene_symbol"] = expr_raw.index.map(probe_to_gene)
    expr_valid = expr_raw.dropna(subset=["gene_symbol"])
    expr_valid = expr_valid[expr_valid["gene_symbol"] != ""]
    
    # Aggregate multiple probes per gene by taking mean
    expr_gene = expr_valid.groupby("gene_symbol").mean()
    
    # Keep only samples in both
    common_samples = meta.index.intersection(expr_gene.columns)
    meta = meta.loc[common_samples]
    expr_gene = expr_gene[common_samples]

    return meta, expr_gene
