"""
GSE132903 Lockbox Cohort Loader (Piras et al. 2019).
Independent middle temporal gyrus (MTG) post-mortem cohort profiled on Illumina HumanHT-12 V4.0.
STRICT LOCKBOX: Used exclusively for zero-leakage external validation.
"""

import os
import re
import pandas as pd
import numpy as np
from src.load.geo_parser import parse_series_matrix, parse_platform_annotation


def load_gse132903(raw_dir="data/raw"):
    """
    Load GSE132903 independent validation cohort.
    Returns:
        meta_df: DataFrame of sample metadata indexed by GSM ID.
        expr_df: DataFrame of gene expression (genes x samples).
    """
    matrix_path = os.path.join(raw_dir, "GSE132903_series_matrix.txt.gz")
    annot_path = os.path.join(raw_dir, "GPL10558.annot.gz")

    if not os.path.exists(matrix_path):
        raise FileNotFoundError(f"GSE132903 series matrix missing at {matrix_path}")
    if not os.path.exists(annot_path):
        raise FileNotFoundError(f"GPL10558 annotation missing at {annot_path}")

    meta_raw, expr_raw = parse_series_matrix(matrix_path, load_expression=True)
    annot = parse_platform_annotation(annot_path)

    meta = pd.DataFrame(index=meta_raw.index)
    meta["sample_id"] = meta_raw.index
    meta["cohort"] = "GSE132903_LOCKBOX"
    meta["platform"] = "GPL10558"
    meta["region"] = "Middle_Temporal_Gyrus"

    # Title format: "middle_temporal_gyrus_ND_1_08-81" -> donor identifier
    meta["donor_id"] = meta_raw["title"].apply(lambda t: str(t).split("_")[-1] if pd.notna(t) else meta_raw.index)

    # Diagnosis: ND (Non-demented / Control), AD (Alzheimer's Disease)
    if "diagnosis" in meta_raw.columns:
        diag = meta_raw["diagnosis"].astype(str).str.strip().str.upper()
        meta["diagnosis"] = diag.map(lambda d: "CTL" if d in ["ND", "CONTROL", "CTL", "NORMAL"] else ("AD" if "AD" in d else d))
    else:
        meta["diagnosis"] = "Unknown"

    # Age
    age_col = None
    for col in meta_raw.columns:
        if "age" in col.lower():
            age_col = col
            break
    if age_col:
        meta["age"] = pd.to_numeric(meta_raw[age_col].astype(str).str.replace("+", "", regex=False), errors="coerce")
    else:
        meta["age"] = np.nan

    # Sex
    sex_col = None
    for col in meta_raw.columns:
        if "sex" in col.lower() or "gender" in col.lower():
            sex_col = col
            break
    if sex_col:
        meta["sex"] = meta_raw[sex_col].astype(str).str.upper().map(lambda s: "FEMALE" if "FEMALE" in s or s == "F" else "MALE")
    else:
        meta["sex"] = "UNKNOWN"

    # Map probes to gene symbols and aggregate by mean
    probe_to_gene = annot.set_index("probe_id")["gene_symbol"].to_dict()
    expr_raw["gene_symbol"] = expr_raw.index.map(probe_to_gene)
    expr_valid = expr_raw.dropna(subset=["gene_symbol"])
    expr_valid = expr_valid[expr_valid["gene_symbol"] != ""]
    expr_gene = expr_valid.groupby("gene_symbol").mean()

    common_samples = meta.index.intersection(expr_gene.columns)
    meta = meta.loc[common_samples]
    expr_gene = expr_gene[common_samples]

    return meta, expr_gene
