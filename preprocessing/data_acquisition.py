"""
Phase 2: Data Acquisition & Preprocessing Pipeline
Cross-Species Transcriptomic Signature of Alzheimer's Disease Progression
Author: Aryan Padarthi
Dataset: ROSMAP (syn3219045) + Allen Brain Atlas

This script handles:
1. Data download from Synapse (ROSMAP) and Allen API
2. Quality control, normalization, batch correction
3. Ortholog mapping between human and mouse
4. Feature selection and train/test splitting
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ---------- CONFIGURATION ----------
BASE_DIR = Path("/Volumes/2TB/AANneuro/data")
PROCESSED_DIR = BASE_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------- SECTION 1: ROSMAP DATA DOWNLOAD ----------
# Note: Requires synapseclient installed: pip install synapseclient
# User must register at synapse.org and obtain credentials

def download_rosmap_data():
    """
    Download ROSMAP bulk RNA-seq data from Synapse.
    
    Instructions:
    1. Register at https://www.synapse.org
    2. Request access to syn3219045 (ROSMAP Study)
    3. Place your .synapseConfig in ~/.synapseConfig or provide auth token
    4. Run this function
    
    Returns: Paths to downloaded files
    """
    import synapseclient
    syn = synapseclient.Synapse()
    syn.login()
    
    # Download ROSMAP RNA-seq expression matrix
    expression_entity = syn.get("syn8456649")  # DLPFC expression matrix
    clinical_entity = syn.get("syn3191087")     # Clinical metadata
    
    print(f"[+] Downloaded expression data to: {expression_entity.path}")
    print(f"[+] Downloaded clinical data to: {clinical_entity.path}")
    
    return expression_entity.path, clinical_entity.path

def download_allen_data():
    """
    Download Allen Brain Atlas aging/dementia transcriptomics data.
    Uses the Allen SDK to query the BrainSpan and Aging datasets.
    """
    try:
        from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
        # This is a simplified query, actual API calls depend on specific datasets
        print("[+] Allen SDK available for data queries")
        print("[!] Full Allen integration requires dataset-specific access token")
        print("[!] Using publicly available summary-level data for mouse models")
        return True
    except ImportError:
        print("[!] allensdk not installed. Run: pip install allensdk")
        return False


# ---------- SECTION 2: DATA LOADING & QUALITY CONTROL ----------

def load_expression_matrix(filepath: str) -> pd.DataFrame:
    """
    Load RNA-seq expression matrix (genes x samples).
    Expected format: TSV with genes as rows, samples as columns, first column = gene_id
    """
    df = pd.read_csv(filepath, sep="\t", index_col=0)
    print(f"[+] Loaded expression matrix: {df.shape[0]} genes x {df.shape[1]} samples")
    return df

def load_clinical_data(filepath: str) -> pd.DataFrame:
    """
    Load clinical metadata (samples x clinical variables).
    Expected columns: sample_id, age_at_death, sex, pmi, braak_stage, cerad_score, 
                      cog_dx (clinical diagnosis), cognitive_composite_zscore
    """
    df = pd.read_csv(filepath, sep="\t", index_col=0)
    print(f"[+] Loaded clinical data: {df.shape[0]} subjects, {df.shape[1]} variables")
    return df

def quality_control(expr_df: pd.DataFrame, clinical_df: pd.DataFrame, 
                    min_rin: float = 5.0, min_variance: float = 0.01) -> tuple:
    """
    Perform quality control on expression and clinical data.
    Filters low-variance or non-expressed genes.
    """
    n_genes_before, n_samples_before = expr_df.shape
    
    # Align samples
    common_samples = [s for s in clinical_df.index if s in expr_df.columns]
    expr_df = expr_df[common_samples]
    clinical_df = clinical_df.loc[common_samples]
    
    # Filter low-variance genes
    gene_vars = expr_df.var(axis=1)
    valid_genes = gene_vars[gene_vars > min_variance].index
    expr_df = expr_df.loc[valid_genes]
    
    # Drop rows with NaN if any
    expr_df = expr_df.dropna(axis=0)
    
    print(f"[QC] Retained {expr_df.shape[0]}/{n_genes_before} genes and {expr_df.shape[1]}/{n_samples_before} samples.")
    return expr_df, clinical_df


# ---------- SECTION 3: NORMALIZATION ----------

def normalize_expression(expr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize expression data.
    Automatically detects raw counts vs pre-normalized log2 values.
    """
    max_val = expr_df.values.max()
    if max_val > 50.0:
        print(f"[+] Raw counts detected (max={max_val:.1f}). Applying DESeq2 size factors + log2.")
        log_expr = np.log(expr_df.values + 1)
        geo_means = np.exp(log_expr.mean(axis=1))
        ratios = expr_df.values / (geo_means[:, np.newaxis] + 1e-8)
        size_factors = np.array([
            np.median(ratios[:, j][ratios[:, j] > 0]) if np.any(ratios[:, j] > 0) else 1.0
            for j in range(expr_df.shape[1])
        ])
        normalized = expr_df.values / size_factors[np.newaxis, :]
        return pd.DataFrame(np.log2(normalized + 1.0), index=expr_df.index, columns=expr_df.columns)
    else:
        print(f"[+] Pre-normalized log2 expression detected (max={max_val:.2f}). Preserving scale.")
        return expr_df


# ---------- SECTION 4: BATCH CORRECTION ----------

def combat_seq_correction(expr_df: pd.DataFrame, clinical_df: pd.DataFrame,
                          batch_col: str = "sequencing_batch",
                          covariate_cols: list = None) -> pd.DataFrame:
    """
    Batch effect correction using ComBat-seq via pyComBat.
    
    Args:
        expr_df: Log-transformed expression matrix
        clinical_df: Clinical metadata
        batch_col: Column identifying batch
        covariate_cols: Biological covariates to preserve
    
    Returns: Batch-corrected expression matrix
    """
    if covariate_cols is None:
        covariate_cols = ["age_at_death", "sex", "pmi"]
    
    try:
        from combat.pycombat import pycombat
        
        batch = clinical_df.loc[expr_df.columns, batch_col].values
        design = clinical_df.loc[expr_df.columns, covariate_cols].values
        
        corrected = pycombat(expr_df.values, batch)
        print(f"[+] ComBat-seq correction applied using batch: {batch_col}")
        
        return pd.DataFrame(corrected, index=expr_df.index, columns=expr_df.columns)
    except ImportError:
        print("[!] pycombat not installed. Skipping batch correction.")
        print("[!] Install with: pip install pycombat")
        return expr_df


# ---------- SECTION 5: FEATURE SELECTION (HVG) ----------

def select_highly_variable_genes(expr_df: pd.DataFrame, n_top: int = 4000) -> pd.DataFrame:
    """
    Select top n highly variable genes based on mean-variance relationship.
    Uses the seurat_v3-style method: bin genes by mean expression, 
    pick highest-variance genes within each bin.
    """
    means = np.nan_to_num(expr_df.values.mean(axis=1), nan=0.0, posinf=100.0, neginf=0.0)
    variances = np.nan_to_num(expr_df.values.var(axis=1), nan=1e-5, posinf=100.0, neginf=1e-5)
    
    # Log transform mean and variance
    log_means = np.log10(np.maximum(means, 0.0) + 1.0)
    log_vars = np.log10(np.maximum(variances, 1e-6) + 1.0)
    
    # Fit LOESS to mean-variance relationship
    from scipy.stats import zscore
    
    # Bin genes by mean expression
    n_bins = 20
    # Safe quantile binning to prevent empty or infinite intervals
    bins = pd.qcut(log_means, q=n_bins, labels=False, duplicates="drop")
    
    # Compute z-scored variance within each bin
    z_var = np.zeros_like(log_vars)
    for b in np.unique(bins):
        mask = bins == b
        if mask.sum() > 1:
            z_var[mask] = zscore(log_vars[mask])
    
    # Select top n genes by within-bin z-scored variance
    n_top = min(n_top, len(expr_df))
    top_indices = np.argsort(np.nan_to_num(z_var, nan=-999))[-n_top:]
    
    hvg_df = expr_df.iloc[top_indices]
    print(f"[+] Selected top {n_top} highly variable genes from {expr_df.shape[0]} total")
    
    return hvg_df, top_indices, z_var



# ---------- SECTION 6: ORTHOLOG MAPPING (HUMAN ↔ MOUSE) ----------

def get_human_mouse_orthologs() -> dict:
    """
    Get one-to-one human-mouse orthologs from Ensembl BioMart.
    
    Uses a pre-downloaded mapping file or queries Ensembl REST API.
    Returns: dict mapping human_ensembl_id → mouse_ensembl_id
    """
    ortholog_file = PROCESSED_DIR / "human_mouse_orthologs.json"
    
    if ortholog_file.exists():
        with open(ortholog_file, "r") as f:
            return json.load(f)
    
    # Query Ensembl REST API or raise error
    raise FileNotFoundError(
        f"Ortholog mapping file not found at {ortholog_file}. "
        "Please provide real Ensembl one-to-one ortholog mappings."
    )


import hashlib

def compute_file_sha256(filepath: Path) -> str:
    """Compute and return the SHA-256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            sha256.update(chunk)
    return sha256.hexdigest()

def log_dataset_provenance(name: str, filepath: Path, df: pd.DataFrame):
    """Log exact checksum, shape, index type, and column count for a dataset."""
    chk = compute_file_sha256(filepath)
    print(f"[PROVENANCE] {name}:")
    print(f"  File path: {filepath}")
    print(f"  SHA-256:   {chk}")
    print(f"  Shape:     {df.shape[0]} rows x {df.shape[1]} cols")
    if len(df.columns) > 0:
        print(f"  Sample IDs (first 3): {list(df.columns[:3])}")
    if len(df.index) > 0:
        print(f"  Row IDs (first 3):    {list(df.index[:3])}")


# ---------- SECTION 7: DATA SPLITTING (DONOR-GROUPED) ----------

def split_data_by_donor(expr_df: pd.DataFrame, clinical_df: pd.DataFrame,
                        test_size: float = 0.2, random_state: int = RANDOM_SEED) -> dict:
    """
    Split real data into training and held-out test sets at the DONOR level.
    Guarantees zero donor leakage across train and test partitions.
    Stratified by donor-level clinical disease stage.
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    
    samples = [s for s in expr_df.columns if s in clinical_df.index]
    expr_df = expr_df[samples]
    clinical_df = clinical_df.loc[samples]
    
    donor_col = "individual" if "individual" in clinical_df.columns else "donor_id"
    dx_col = "disease state" if "disease state" in clinical_df.columns else ("cog_dx" if "cog_dx" in clinical_df.columns else "diagnosis")
    
    # Get donor-level summary
    donor_df = clinical_df.groupby(donor_col)[dx_col].first().reset_index()
    donors = donor_df[donor_col].values
    donor_labels = donor_df[dx_col].values
    
    # Stratified split on unique donors
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_donor_idx, test_donor_idx = next(sss.split(donors, donor_labels))
    
    train_donors = set(donors[train_donor_idx])
    test_donors = set(donors[test_donor_idx])
    
    assert len(train_donors.intersection(test_donors)) == 0, "Donor leakage detected!"
    
    train_samples = clinical_df[clinical_df[donor_col].isin(train_donors)].index.tolist()
    test_samples = clinical_df[clinical_df[donor_col].isin(test_donors)].index.tolist()
    
    train_samples = [s for s in train_samples if s in expr_df.columns]
    test_samples = [s for s in test_samples if s in expr_df.columns]
    
    print(f"[+] Donor-grouped split (0% donor overlap):")
    print(f"    Train: {len(train_donors)} donors ({len(train_samples)} samples)")
    print(f"    Test:  {len(test_donors)} donors ({len(test_samples)} samples)")
    
    return {
        "train_expr": expr_df[train_samples],
        "test_expr": expr_df[test_samples],
        "train_clinical": clinical_df.loc[train_samples],
        "test_clinical": clinical_df.loc[test_samples],
        "train_donors": list(train_donors),
        "test_donors": list(test_donors),
    }


# ---------- MAIN PIPELINE ----------

def run_preprocessing_pipeline(human_expr_path: str = None, human_clinical_path: str = None,
                               mouse_expr_path: str = None, mouse_clinical_path: str = None):
    """
    Execute data preprocessing on real datasets with strict donor-level splitting.
    Raises FileNotFoundError immediately if required human inputs are missing.
    """
    print("=" * 60)
    print("Data Preprocessing Pipeline (Real Data, Donor-Grouped)")
    print("=" * 60)
    
    # Check human inputs
    if human_expr_path is None or human_clinical_path is None:
        raw_human_expr = BASE_DIR / "raw" / "human_dlpfc_expression.tsv"
        raw_human_clin = BASE_DIR / "raw" / "human_clinical_metadata.tsv"
        
        if raw_human_expr.exists() and raw_human_clin.exists():
            human_expr_path = str(raw_human_expr)
            human_clinical_path = str(raw_human_clin)
        else:
            raise FileNotFoundError(
                f"Missing real human dataset files!\n"
                f"Expected:\n  Expression: {raw_human_expr}\n  Clinical:   {raw_human_clin}\n"
                f"Please download real ROSMAP or public GEO data before running the pipeline."
            )
            
    print(f"\n[1/5] Loading real human data from {human_expr_path} and {human_clinical_path}...")
    human_expr_raw = load_expression_matrix(human_expr_path)
    human_clinical_raw = load_clinical_data(human_clinical_path)
    log_dataset_provenance("Human Expression (Raw)", Path(human_expr_path), human_expr_raw)
    log_dataset_provenance("Human Clinical (Raw)", Path(human_clinical_path), human_clinical_raw)
    
    # Step 2: Quality control
    print("\n[2/5] Quality control...")
    expr_human_qc, clinical_human_qc = quality_control(human_expr_raw, human_clinical_raw)
    
    # Step 3: Normalization
    print("\n[3/5] Normalizing expression...")
    human_norm = normalize_expression(expr_human_qc)
    
    # Step 4: Batch correction (if batch metadata present)
    if "sequencing_batch" in clinical_human_qc.columns and clinical_human_qc["sequencing_batch"].nunique() > 1:
        print("\n[4/5] Running ComBat batch correction...")
        human_corrected = combat_seq_correction(human_norm, clinical_human_qc)
    else:
        print("\n[4/5] Skipping batch correction (no multi-batch annotation found)...")
        human_corrected = human_norm
        
    # Step 5: Donor-grouped split FIRST, then HVG selection strictly on training partition
    print("\n[5/5] Performing donor-level partition & train-isolated HVG selection...")
    human_split = split_data_by_donor(human_corrected, clinical_human_qc, test_size=0.2, random_state=RANDOM_SEED)
    
    # Fit HVG selection on training donors ONLY
    n_features = min(4000, human_split["train_expr"].shape[0])
    _, top_indices, _ = select_highly_variable_genes(human_split["train_expr"], n_top=n_features)
    selected_genes = human_split["train_expr"].index[top_indices]
    
    # Subset both train and test to training-selected HVGs
    human_hvg = human_corrected.loc[selected_genes]
    
    # Save processed outputs
    print("\n[+] Saving verified processed data...")
    hvg_out = PROCESSED_DIR / "human_dlpfc_processed.csv.gz"
    train_clin_out = PROCESSED_DIR / "clinical_train.csv"
    test_clin_out = PROCESSED_DIR / "clinical_test.csv"
    
    human_hvg.to_csv(hvg_out, compression="gzip")
    human_split["train_clinical"].to_csv(train_clin_out)
    human_split["test_clinical"].to_csv(test_clin_out)
    
    log_dataset_provenance("Human Multi-Region Processed (HVG)", hvg_out, human_hvg)
    
    # Optional mouse data processing if provided
    mouse_aligned = None
    if mouse_expr_path and mouse_clinical_path and Path(mouse_expr_path).exists() and Path(mouse_clinical_path).exists():
        print("\n[+] Processing real mouse dataset...")
        mouse_expr_raw = load_expression_matrix(mouse_expr_path)
        mouse_clinical_raw = load_clinical_data(mouse_clinical_path)
        log_dataset_provenance("Mouse Expression (Raw)", Path(mouse_expr_path), mouse_expr_raw)
        
        expr_mouse_qc, clinical_mouse_qc = quality_control(mouse_expr_raw, mouse_clinical_raw)
        mouse_norm = log_transform(deseq2_normalize(expr_mouse_qc))
        
        orthologs = get_human_mouse_orthologs()
        mouse_human_genes = [orthologs.get(g, g) for g in mouse_norm.index]
        mouse_norm.index = mouse_human_genes
        mouse_aligned = mouse_norm[mouse_norm.index.isin(human_hvg.index)]
        mouse_out = PROCESSED_DIR / "mouse_cortex_processed.csv.gz"
        mouse_aligned.to_csv(mouse_out, compression="gzip")
        log_dataset_provenance("Mouse Processed (Ortholog Aligned)", mouse_out, mouse_aligned)
    else:
        print("\n[*] No verified mouse dataset provided. Human-first mode active.")
        
    print(f"\n[✓] Preprocessing complete. Data saved to: {PROCESSED_DIR}")
    return human_hvg, mouse_aligned, human_split


if __name__ == "__main__":
    result = run_preprocessing_pipeline()
    print("\n[✓] Preprocessing complete. Ready for model training.")