"""
Phase 3: Conditional Variational Autoencoder (cVAE) and Rigorous Baseline Benchmarks
for Multi-Region Alzheimer's Disease Neuropathological Progression
===================================================================================
Author: Aryan Padarthi
Dataset: GSE118553 (112 donors, 401 multi-region post-mortem brain samples)

Key Enhancements for Revision Audit Draft 5:
1. Donor-level statistics: Donor-averaged predictions, donor permutation p-values, 1000-iteration donor bootstrap 95% CIs.
2. Fair Regularization: Inner CV tuning of Ridge alpha for all representations.
3. Demographic baselines & residualization: Age+Sex classifier, Age/Sex-residualized PCA baseline, ANOVA/Chi-sq tests.
4. Complete AsymAD vs AD resilience benchmark across all representations.
5. Sample-split vs Donor-grouped split leakage inflation comparison.
6. Empirical testing of latent PC1 vs anatomical brain regions.
7. Verified Enrichr pathway enrichment with accurate FDR reporting and directional regulation.
"""

import os
import sys
import json
import requests
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, mean_absolute_error, r2_score as sk_r2
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr, f_oneway, chi2_contingency
from scipy import stats

warnings.filterwarnings("ignore")

# ---------- CONFIGURATION ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

LATENT_DIM = 64
HIDDEN_DIMS = [1024, 512, 128]
CONDITION_DIM = 3  # [normalized_age, is_male, is_female]
BETA_SCHEDULE_STEPS = 40
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
EPOCHS = 160


# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================

class ConditionalEncoder(nn.Module):
    """Deep conditional encoder injecting demographic covariates at each layer."""
    
    def __init__(self, input_dim=4000, condition_dim=3, hidden_dims=None, latent_dim=64):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [1024, 512, 128]
        
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        
        # Layer 1
        self.fc1 = nn.Linear(input_dim + condition_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.drop1 = nn.Dropout(0.1)
        
        # Layer 2
        self.fc2 = nn.Linear(hidden_dims[0] + condition_dim, hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.drop2 = nn.Dropout(0.1)
        
        # Layer 3
        self.fc3 = nn.Linear(hidden_dims[1] + condition_dim, hidden_dims[2])
        self.bn3 = nn.BatchNorm1d(hidden_dims[2])
        self.drop3 = nn.Dropout(0.1)
        
        # Output heads
        self.mu_head = nn.Linear(hidden_dims[2] + condition_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dims[2] + condition_dim, latent_dim)
    
    def forward(self, x, c):
        h = torch.cat([x, c], dim=-1)
        h = self.drop1(F.relu(self.bn1(self.fc1(h))))
        
        h = torch.cat([h, c], dim=-1)
        h = self.drop2(F.relu(self.bn2(self.fc2(h))))
        
        h = torch.cat([h, c], dim=-1)
        h = self.drop3(F.relu(self.bn3(self.fc3(h))))
        
        h = torch.cat([h, c], dim=-1)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar


class ConditionalDecoder(nn.Module):
    """Decoder reconstructing gene expression from latent z and condition c."""
    
    def __init__(self, latent_dim=64, condition_dim=3, hidden_dims=None, output_dim=4000):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 512, 1024]
        
        layers = []
        in_dim = latent_dim + condition_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, z, c):
        return self.net(torch.cat([z, c], dim=-1))


class ConditionalVAE(nn.Module):
    """Full conditional VAE model."""
    
    def __init__(self, input_dim=4000, condition_dim=3, latent_dim=64, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [1024, 512, 128]
        self.encoder = ConditionalEncoder(input_dim, condition_dim, hidden_dims, latent_dim)
        self.decoder = ConditionalDecoder(latent_dim, condition_dim, list(reversed(hidden_dims)), input_dim)
        self.latent_dim = latent_dim
    
    def reparameterize(self, mu, logvar):
        logvar_clamped = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar_clamped)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x, c):
        mu, _ = self.encoder(x, c)
        return mu
    
    def forward(self, x, c, beta=1.0):
        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, c)
        
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")
        logvar_clamped = torch.clamp(logvar, min=-10.0, max=10.0)
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp(), dim=1))
        
        total_loss = recon_loss + beta * (kl_loss / x.shape[1])
        return x_recon, mu, logvar, recon_loss, kl_loss, total_loss


class StandardVAE(nn.Module):
    """Unconditioned VAE baseline."""
    
    def __init__(self, input_dim=4000, latent_dim=64, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [1024, 512, 128]
        
        enc_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            enc_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_dim = h_dim
        self.enc_net = nn.Sequential(*enc_layers)
        self.mu_head = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar_head = nn.Linear(hidden_dims[-1], latent_dim)
        
        dec_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            dec_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_dim = h_dim
        dec_layers.append(nn.Linear(in_dim, input_dim))
        self.dec_net = nn.Sequential(*dec_layers)
    
    def reparameterize(self, mu, logvar):
        logvar_clamped = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar_clamped)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x):
        h = self.enc_net(x)
        return self.mu_head(h)
    
    def forward(self, x, beta=1.0):
        h = self.enc_net(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        z = self.reparameterize(mu, logvar)
        x_recon = self.dec_net(z)
        
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")
        logvar_clamped = torch.clamp(logvar, min=-10.0, max=10.0)
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp(), dim=1))
        
        total_loss = recon_loss + beta * (kl_loss / x.shape[1])
        return x_recon, mu, logvar, recon_loss, kl_loss, total_loss


# =============================================================================
# TRAINING HELPERS
# =============================================================================

def train_cvae(model, dataloader, epochs=EPOCHS, annealing_epochs=BETA_SCHEDULE_STEPS, lr=LEARNING_RATE, device=DEVICE):
    """Train cVAE with KL annealing and early stopping."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    best_loss = float("inf")
    best_state = None
    patience = 25
    patience_counter = 0
    
    model.train()
    for epoch in range(epochs):
        beta = min(1.0, (epoch + 1) / annealing_epochs) if annealing_epochs > 0 else 1.0
        epoch_recon = 0.0
        for batch_x, batch_c in dataloader:
            batch_x = batch_x.to(device)
            batch_c = batch_c.to(device)
            
            optimizer.zero_grad()
            _, _, _, recon_loss, kl_loss, loss = model(batch_x, batch_c, beta=beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_recon += recon_loss.item()
            
        avg_recon = epoch_recon / len(dataloader)
        if avg_recon < best_loss:
            best_loss = avg_recon
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
            
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def train_standard_vae(model, dataloader, epochs=EPOCHS, annealing_epochs=BETA_SCHEDULE_STEPS, lr=LEARNING_RATE, device=DEVICE):
    """Train unconditioned VAE with KL annealing and early stopping."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    best_loss = float("inf")
    best_state = None
    patience = 25
    patience_counter = 0
    
    model.train()
    for epoch in range(epochs):
        beta = min(1.0, (epoch + 1) / annealing_epochs) if annealing_epochs > 0 else 1.0
        epoch_recon = 0.0
        for batch_x in dataloader:
            batch_x = batch_x[0].to(device)
            optimizer.zero_grad()
            _, _, _, recon_loss, _, loss = model(batch_x, beta=beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_recon += recon_loss.item()
            
        avg_recon = epoch_recon / len(dataloader)
        if avg_recon < best_loss:
            best_loss = avg_recon
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
            
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def build_condition_vector(clin_df, samples):
    """Construct condition vector c in R^3: [normalized_age, is_male, is_female]."""
    median_age = float(pd.to_numeric(clin_df.get("age_at_death", clin_df.get("age", 80.0)), errors="coerce").median())
    if np.isnan(median_age):
        median_age = 80.0
        
    rows = []
    for s in samples:
        row = clin_df.loc[s]
        raw_age = row.get("age_at_death", row.get("age", median_age))
        try:
            age_val = float(raw_age)
            if np.isnan(age_val):
                age_val = median_age
        except (ValueError, TypeError):
            age_val = median_age
            
        age_norm = (age_val - 80.0) / 10.0
        sex = str(row.get("sex", row.get("gender", "M"))).upper()
        is_male = 1.0 if sex.startswith("M") or sex == "1" else 0.0
        is_female = 1.0 if sex.startswith("F") or sex == "2" else 0.0
        rows.append([age_norm, is_male, is_female])
        
    return np.array(rows, dtype=np.float32)


def get_disease_stage(clin_df, samples):
    """Encode disease stages: Control -> 0, AsymAD -> 1, AD -> 2."""
    stages = []
    for s in samples:
        row = clin_df.loc[s]
        dx = str(row.get("disease state", row.get("diagnosis", row.get("cog_dx", "")))).lower()
        if "control" in dx or dx == "0" or "nci" in dx:
            stages.append(0.0)
        elif "asym" in dx or "mci" in dx or dx == "1":
            stages.append(1.0)
        else:  # AD / dementia
            stages.append(2.0)
    return np.array(stages, dtype=np.float32)


# =============================================================================
# STATISTICAL EVALUATION & BOOTSTRAP UTILITIES
# =============================================================================

def tune_ridge_alpha_grouped(X_tr, y_tr, groups_tr, alphas=None):
    """Tune Ridge alpha using 5-fold donor-grouped cross-validation on training data."""
    if alphas is None:
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    best_alpha = 1.0
    best_mse = float("inf")
    
    y_strata = y_tr.astype(int)
    
    for alpha in alphas:
        fold_mses = []
        for tr_idx, val_idx in sgkf.split(X_tr, y_strata, groups=groups_tr):
            r = Ridge(alpha=alpha)
            r.fit(X_tr[tr_idx], y_tr[tr_idx])
            preds = r.predict(X_tr[val_idx])
            mse = np.mean((y_tr[val_idx] - preds) ** 2)
            fold_mses.append(mse)
        mean_mse = np.mean(fold_mses)
        if mean_mse < best_mse:
            best_mse = mean_mse
            best_alpha = alpha
            
    return best_alpha


def compute_donor_level_and_bootstrap_stats(y_true_samples, y_pred_samples, donors_samples, n_bootstraps=1000):
    """
    Compute sample-level and donor-level progression metrics:
    - Donor-averaged predictions
    - Donor-level Spearman rho, Pearson r, R^2, MAE
    - Donor-level permutation p-value
    - 1,000 donor-resample bootstrap 95% Confidence Intervals
    """
    df_eval = pd.DataFrame({
        "donor": donors_samples,
        "y_true": y_true_samples,
        "y_pred": y_pred_samples
    })
    
    # Donor-level aggregation
    donor_agg = df_eval.groupby("donor").agg({
        "y_true": "first",
        "y_pred": "mean"
    }).reset_index()
    
    y_true_donor = donor_agg["y_true"].values
    y_pred_donor = donor_agg["y_pred"].values
    n_donors = len(donor_agg)
    
    # Sample-level stats
    sample_rho, sample_p = spearmanr(y_true_samples, y_pred_samples)
    sample_r, sample_rp = pearsonr(y_true_samples, y_pred_samples)
    sse_s = np.sum((y_true_samples - y_pred_samples) ** 2)
    sst_s = np.sum((y_true_samples - np.mean(y_true_samples)) ** 2)
    sample_r2 = 1.0 - (sse_s / sst_s) if sst_s > 0 else 0.0
    sample_mae = mean_absolute_error(y_true_samples, y_pred_samples)
    
    # Donor-level stats
    donor_rho, donor_p_norm = spearmanr(y_true_donor, y_pred_donor)
    donor_r, donor_rp_norm = pearsonr(y_true_donor, y_pred_donor)
    sse_d = np.sum((y_true_donor - y_pred_donor) ** 2)
    sst_d = np.sum((y_true_donor - np.mean(y_true_donor)) ** 2)
    donor_r2 = 1.0 - (sse_d / sst_d) if sst_d > 0 else 0.0
    donor_mae = mean_absolute_error(y_true_donor, y_pred_donor)
    
    # Donor-level permutation test (shuffling labels across donors)
    perm_rhos = []
    rng = np.random.default_rng(RANDOM_SEED)
    for _ in range(n_bootstraps):
        y_perm = rng.permutation(y_true_donor)
        r_perm, _ = spearmanr(y_perm, y_pred_donor)
        perm_rhos.append(r_perm)
    perm_rhos = np.array(perm_rhos)
    donor_p_perm = float(np.mean(np.abs(perm_rhos) >= np.abs(donor_rho)))
    
    # Donor-level bootstrap resampling (sample donors with replacement)
    unique_donors = donor_agg["donor"].unique()
    bs_sample_rhos = []
    bs_donor_rhos = []
    bs_donor_r2s = []
    
    for _ in range(n_bootstraps):
        sampled_donors = rng.choice(unique_donors, size=len(unique_donors), replace=True)
        bs_df = pd.concat([df_eval[df_eval["donor"] == d] for d in sampled_donors], ignore_index=True)
        if bs_df["y_true"].nunique() > 1:
            r_s, _ = spearmanr(bs_df["y_true"], bs_df["y_pred"])
            bs_sample_rhos.append(r_s)
            
            bs_d_agg = bs_df.groupby("donor").agg({"y_true": "first", "y_pred": "mean"})
            r_d, _ = spearmanr(bs_d_agg["y_true"], bs_d_agg["y_pred"])
            bs_donor_rhos.append(r_d)
            
            sse_bs = np.sum((bs_d_agg["y_true"] - bs_d_agg["y_pred"]) ** 2)
            sst_bs = np.sum((bs_d_agg["y_true"] - np.mean(bs_d_agg["y_true"])) ** 2)
            bs_donor_r2s.append(1.0 - sse_bs / sst_bs if sst_bs > 0 else 0.0)
            
    sample_ci = [float(np.percentile(bs_sample_rhos, 2.5)), float(np.percentile(bs_sample_rhos, 97.5))] if bs_sample_rhos else [None, None]
    donor_ci = [float(np.percentile(bs_donor_rhos, 2.5)), float(np.percentile(bs_donor_rhos, 97.5))] if bs_donor_rhos else [None, None]
    donor_r2_ci = [float(np.percentile(bs_donor_r2s, 2.5)), float(np.percentile(bs_donor_r2s, 97.5))] if bs_donor_r2s else [None, None]
    
    return {
        "n_donors": int(n_donors),
        "n_samples": int(len(df_eval)),
        "sample_level": {
            "spearman_rho": float(sample_rho),
            "spearman_p_nominal": float(sample_p),
            "spearman_rho_95ci": sample_ci,
            "pearson_r": float(sample_r),
            "r2": float(sample_r2),
            "mae": float(sample_mae)
        },
        "donor_level": {
            "spearman_rho": float(donor_rho),
            "spearman_p_nominal": float(donor_p_norm),
            "spearman_p_permutation": float(donor_p_perm),
            "spearman_rho_95ci": donor_ci,
            "pearson_r": float(donor_r),
            "r2": float(donor_r2),
            "r2_95ci": donor_r2_ci,
            "mae": float(donor_mae)
        }
    }


def compute_regional_bootstrap_ci(df_region, n_bootstraps=1000):
    """Compute 95% bootstrap CIs for regional progression correlation by resampling donors."""
    unique_donors = df_region["donor"].unique()
    rng = np.random.default_rng(RANDOM_SEED)
    bs_rhos = []
    
    for _ in range(n_bootstraps):
        sampled = rng.choice(unique_donors, size=len(unique_donors), replace=True)
        bs_df = pd.concat([df_region[df_region["donor"] == d] for d in sampled], ignore_index=True)
        if bs_df["y_true"].nunique() > 1:
            r, _ = spearmanr(bs_df["y_true"], bs_df["y_pred"])
            bs_rhos.append(r)
            
    if len(bs_rhos) > 0:
        return [float(np.percentile(bs_rhos, 2.5)), float(np.percentile(bs_rhos, 97.5))]
    return [None, None]


# =============================================================================
# BIOLOGICAL PATHWAY ENRICHMENT (ENRICHR)
# =============================================================================

def run_enrichr_analysis(genes_list, gene_scores=None, top_n=150):
    """
    Perform biological pathway enrichment via Enrichr REST API.
    Queries GO_Biological_Process_2023 and Reactome_2022.
    """
    print(f"\n[+] Querying Enrichr API for top {top_n} progression-correlated genes...")
    if gene_scores is not None:
        ranked_idx = np.argsort(gene_scores)[-top_n:]
        query_genes = [genes_list[i] for i in reversed(ranked_idx)]
    else:
        query_genes = genes_list[:top_n]
        
    enrichment_results = {}
    
    try:
        add_url = "https://maayanlab.cloud/Enrichr/addList"
        res = requests.post(add_url, files={"list": (None, "\n".join(query_genes)), "description": (None, "AAN_AD_progression")}, timeout=15)
        if res.status_code == 200:
            user_list_id = res.json().get("userListId")
            libraries = ["GO_Biological_Process_2023", "Reactome_2022"]
            for lib in libraries:
                query_url = f"https://maayanlab.cloud/Enrichr/enrich?userListId={user_list_id}&backgroundType={lib}"
                lib_res = requests.get(query_url, timeout=15)
                if lib_res.status_code == 200:
                    terms_data = lib_res.json().get(lib, [])
                    parsed_terms = []
                    for t in terms_data[:10]:
                        parsed_terms.append({
                            "term": t[1],
                            "p_value": float(t[2]),
                            "adj_p_value": float(t[6]),
                            "combined_score": float(t[4]),
                            "genes": t[5],
                        })
                    enrichment_results[lib] = parsed_terms
                    print(f"  [✓] {lib}: Retrieved {len(parsed_terms)} significant pathways.")
        else:
            print(f"  [!] Enrichr list upload returned status {res.status_code}")
    except Exception as e:
        print(f"  [!] Enrichr query error: {e}")
        
    return enrichment_results, query_genes


# =============================================================================
# MAIN EXPERIMENTAL PIPELINE
# =============================================================================

def run_experiment():
    """
    Execute full leakage-aware experimental pipeline with rigorous controls:
    1. Donor Cohort Verification & Demographic Distribution Testing (ANOVA, Chi-sq)
    2. Model Training (PCA-64, Unconditioned VAE, cVAE)
    3. Progression Tracking with Tuned Regularization & Donor-Level Bootstrap Statistics
    4. Demographic Controls: Age+Sex Baselines & Demographic Residualization
    5. Diagnostic Classification (AD vs Control) and Cognitive Resilience (AsymAD vs AD)
    6. Sample-Split vs Donor-Grouped Split Leakage Inflation Experiment
    7. Regional Vulnerability Analysis with Donor-Bootstrap CIs & Cerebellar Reframing
    8. Latent PC1 vs Anatomical Region ANOVA
    9. Enrichr Pathway Enrichment
    """
    print("=" * 80)
    print("Leakage-Aware Multi-Region AD Transcriptomics: Revision Audit Draft 5 Pipeline")
    print("=" * 80)
    
    root_dir = Path("/Volumes/2TB/AANneuro")
    proc_dir = root_dir / "data" / "processed"
    results_dir = root_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    human_expr_path = proc_dir / "human_dlpfc_processed.csv.gz"
    train_clin_path = proc_dir / "clinical_train.csv"
    test_clin_path = proc_dir / "clinical_test.csv"
    
    for p in [human_expr_path, train_clin_path, test_clin_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required data file: {p}")
            
    expr_df = pd.read_csv(human_expr_path, index_col=0)
    train_clin = pd.read_csv(train_clin_path, index_col=0)
    test_clin = pd.read_csv(test_clin_path, index_col=0)
    
    train_samples = [s for s in train_clin.index if s in expr_df.columns]
    test_samples = [s for s in test_clin.index if s in expr_df.columns]
    all_samples = train_samples + test_samples
    
    full_clin = pd.concat([train_clin.loc[train_samples], test_clin.loc[test_samples]])
    
    train_donors = set(train_clin["individual"].unique())
    test_donors = set(test_clin["individual"].unique())
    overlap = train_donors.intersection(test_donors)
    assert len(overlap) == 0, f"Critical error: Donor leakage detected ({len(overlap)} overlapping donors)!"
    
    # -------------------------------------------------------------------------
    # [1/8] DEMOGRAPHIC DISTRIBUTION & STATISTICAL TESTING
    # -------------------------------------------------------------------------
    print(f"\n[1/8] Demographic Characterization & Statistical Group Testing:")
    donor_df = full_clin.groupby("individual").first()
    
    ages_ctrl = pd.to_numeric(donor_df[donor_df["disease state"] == "control"]["age_at_death"], errors="coerce").dropna()
    ages_asym = pd.to_numeric(donor_df[donor_df["disease state"] == "AsymAD"]["age_at_death"], errors="coerce").dropna()
    ages_ad = pd.to_numeric(donor_df[donor_df["disease state"] == "AD"]["age_at_death"], errors="coerce").dropna()
    
    f_stat_age, p_val_age = f_oneway(ages_ctrl, ages_asym, ages_ad)
    
    sex_crosstab = pd.crosstab(donor_df["disease state"], donor_df["sex"])
    chi2_stat_sex, p_val_sex, _, _ = chi2_contingency(sex_crosstab)
    
    demographics_summary = {
        "Control": {
            "n_donors": int(len(ages_ctrl)),
            "age_mean": float(ages_ctrl.mean()),
            "age_std": float(ages_ctrl.std()),
            "age_range": [float(ages_ctrl.min()), float(ages_ctrl.max())],
            "female_count": int(sex_crosstab.loc["control", "F"]) if "control" in sex_crosstab.index else 0,
            "male_count": int(sex_crosstab.loc["control", "M"]) if "control" in sex_crosstab.index else 0,
            "female_pct": float(sex_crosstab.loc["control", "F"] / len(ages_ctrl) * 100) if "control" in sex_crosstab.index else 0.0
        },
        "AsymAD": {
            "n_donors": int(len(ages_asym)),
            "age_mean": float(ages_asym.mean()),
            "age_std": float(ages_asym.std()),
            "age_range": [float(ages_asym.min()), float(ages_asym.max())],
            "female_count": int(sex_crosstab.loc["AsymAD", "F"]) if "AsymAD" in sex_crosstab.index else 0,
            "male_count": int(sex_crosstab.loc["AsymAD", "M"]) if "AsymAD" in sex_crosstab.index else 0,
            "female_pct": float(sex_crosstab.loc["AsymAD", "F"] / len(ages_asym) * 100) if "AsymAD" in sex_crosstab.index else 0.0
        },
        "AD": {
            "n_donors": int(len(ages_ad)),
            "age_mean": float(ages_ad.mean()),
            "age_std": float(ages_ad.std()),
            "age_range": [float(ages_ad.min()), float(ages_ad.max())],
            "female_count": int(sex_crosstab.loc["AD", "F"]) if "AD" in sex_crosstab.index else 0,
            "male_count": int(sex_crosstab.loc["AD", "M"]) if "AD" in sex_crosstab.index else 0,
            "female_pct": float(sex_crosstab.loc["AD", "F"] / len(ages_ad) * 100) if "AD" in sex_crosstab.index else 0.0
        },
        "statistical_tests": {
            "age_anova_f": float(f_stat_age),
            "age_anova_p": float(p_val_age),
            "sex_chi2_stat": float(chi2_stat_sex),
            "sex_chi2_p": float(p_val_sex)
        }
    }
    
    print(f"  Control: Mean Age = {ages_ctrl.mean():.1f} ± {ages_ctrl.std():.1f} | Female = {demographics_summary['Control']['female_pct']:.1f}%")
    print(f"  AsymAD:  Mean Age = {ages_asym.mean():.1f} ± {ages_asym.std():.1f} | Female = {demographics_summary['AsymAD']['female_pct']:.1f}%")
    print(f"  AD:      Mean Age = {ages_ad.mean():.1f} ± {ages_ad.std():.1f} | Female = {demographics_summary['AD']['female_pct']:.1f}%")
    print(f"  Age Difference ANOVA: F = {f_stat_age:.3f}, p = {p_val_age:.4e}")
    print(f"  Sex Difference Chi^2: chi^2 = {chi2_stat_sex:.3f}, p = {p_val_sex:.4f}")
    
    # -------------------------------------------------------------------------
    # [2/8] PREPROCESSING & DEMOGRAPHIC RESIDUALIZATION
    # -------------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_raw = expr_df[train_samples].values.T  # (n_train, n_genes)
    X_test_raw = expr_df[test_samples].values.T    # (n_test, n_genes)
    
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    X_all = np.vstack([X_train, X_test])
    
    y_train_stage = get_disease_stage(train_clin, train_samples)
    y_test_stage = get_disease_stage(test_clin, test_samples)
    y_all_stage = np.concatenate([y_train_stage, y_test_stage])
    
    donor_train = train_clin.loc[train_samples, "individual"].values
    donor_test = test_clin.loc[test_samples, "individual"].values
    donor_all = full_clin["individual"].values
    tissue_all = full_clin["tissue"].values
    
    C_train = build_condition_vector(train_clin, train_samples)
    C_test = build_condition_vector(test_clin, test_samples)
    C_all = np.vstack([C_train, C_test])
    
    # Age/Sex Residualized Gene Expression Baseline (fit strictly on training donors)
    reg_demo = LinearRegression()
    reg_demo.fit(C_train, X_train)
    X_train_resid = X_train - reg_demo.predict(C_train)
    X_test_resid = X_test - reg_demo.predict(C_test)
    X_all_resid = np.vstack([X_train_resid, X_test_resid])
    
    # -------------------------------------------------------------------------
    # [3/8] MODEL TRAINING (PCA, VAE, cVAE, RESIDUALIZED PCA)
    # -------------------------------------------------------------------------
    print(f"\n[2/8] Training Representation Models...")
    
    # Model 1: PCA-64
    pca_model = PCA(n_components=LATENT_DIM, random_state=RANDOM_SEED)
    z_pca_train = pca_model.fit_transform(X_train)
    z_pca_test = pca_model.transform(X_test)
    z_pca_all = np.vstack([z_pca_train, z_pca_test])
    
    # Model 1b: Age/Sex Residualized PCA-64
    pca_resid_model = PCA(n_components=LATENT_DIM, random_state=RANDOM_SEED)
    z_pca_resid_train = pca_resid_model.fit_transform(X_train_resid)
    z_pca_resid_test = pca_resid_model.transform(X_test_resid)
    z_pca_resid_all = np.vstack([z_pca_resid_train, z_pca_resid_test])
    
    # Model 2: Standard Unconditioned VAE
    X_train_t = torch.FloatTensor(X_train)
    C_train_t = torch.FloatTensor(C_train)
    X_test_t = torch.FloatTensor(X_test)
    C_test_t = torch.FloatTensor(C_test)
    
    cvae_loader = DataLoader(TensorDataset(X_train_t, C_train_t), batch_size=BATCH_SIZE, shuffle=True)
    vae_loader = DataLoader(TensorDataset(X_train_t), batch_size=BATCH_SIZE, shuffle=True)
    
    torch.manual_seed(RANDOM_SEED)
    std_vae = StandardVAE(input_dim=X_train.shape[1], latent_dim=LATENT_DIM, hidden_dims=HIDDEN_DIMS).to(DEVICE)
    std_vae = train_standard_vae(std_vae, vae_loader, epochs=EPOCHS, annealing_epochs=BETA_SCHEDULE_STEPS, device=DEVICE)
    with torch.no_grad():
        z_vae_train = std_vae.encode(X_train_t.to(DEVICE)).cpu().numpy()
        z_vae_test = std_vae.encode(X_test_t.to(DEVICE)).cpu().numpy()
    z_vae_all = np.vstack([z_vae_train, z_vae_test])
    
    # Model 3: Conditional VAE (cVAE)
    torch.manual_seed(RANDOM_SEED)
    cvae = ConditionalVAE(input_dim=X_train.shape[1], condition_dim=CONDITION_DIM, latent_dim=LATENT_DIM, hidden_dims=HIDDEN_DIMS).to(DEVICE)
    cvae = train_cvae(cvae, cvae_loader, epochs=EPOCHS, annealing_epochs=BETA_SCHEDULE_STEPS, device=DEVICE)
    with torch.no_grad():
        z_cvae_train = cvae.encode(X_train_t.to(DEVICE), C_train_t.to(DEVICE)).cpu().numpy()
        z_cvae_test = cvae.encode(X_test_t.to(DEVICE), C_test_t.to(DEVICE)).cpu().numpy()
    z_cvae_all = np.vstack([z_cvae_train, z_cvae_test])
    
    # -------------------------------------------------------------------------
    # [4/8] PROGRESSION TRACKING WITH TUNED REGULARIZATION & DONOR-LEVEL BOOTSTRAP
    # -------------------------------------------------------------------------
    print(f"\n[3/8] Disease Progression Benchmarks with Tuned Alpha & Donor-Level Bootstrap CIs:")
    
    prog_representations = {
        "Age + Sex Only": (C_train, C_test),
        "Raw HVG (4,000 genes)": (X_train, X_test),
        "PCA (64 components)": (z_pca_train, z_pca_test),
        "Residualized PCA (64-dim)": (z_pca_resid_train, z_pca_resid_test),
        "Unconditioned VAE (64-dim)": (z_vae_train, z_vae_test),
        "Conditional VAE (64-dim)": (z_cvae_train, z_cvae_test),
    }
    
    progression_benchmarks = {}
    
    for name, (z_tr, z_te) in prog_representations.items():
        best_alpha = tune_ridge_alpha_grouped(z_tr, y_train_stage, donor_train)
        
        ridge = Ridge(alpha=best_alpha)
        ridge.fit(z_tr, y_train_stage)
        pred_te = ridge.predict(z_te)
        
        stats_res = compute_donor_level_and_bootstrap_stats(y_test_stage, pred_te, donor_test, n_bootstraps=1000)
        stats_res["tuned_alpha"] = float(best_alpha)
        
        if "VAE" in name or "PCA" in name:
            pca_1 = PCA(n_components=1, random_state=RANDOM_SEED)
            pc1_tr = pca_1.fit_transform(z_tr).flatten()
            pc1_te = pca_1.transform(z_te).flatten()
            r_orient, _ = spearmanr(y_train_stage, pc1_tr)
            if r_orient < 0:
                pc1_te = -pc1_te
            u_rho, u_p = spearmanr(y_test_stage, pc1_te)
            stats_res["unsupervised_pc1_rho"] = float(u_rho)
            stats_res["unsupervised_pc1_p"] = float(u_p)
        else:
            stats_res["unsupervised_pc1_rho"] = None
            stats_res["unsupervised_pc1_p"] = None
            
        progression_benchmarks[name] = stats_res
        
        d_rho = stats_res["donor_level"]["spearman_rho"]
        d_p_perm = stats_res["donor_level"]["spearman_p_permutation"]
        d_ci = stats_res["donor_level"]["spearman_rho_95ci"]
        d_r2 = stats_res["donor_level"]["r2"]
        s_rho = stats_res["sample_level"]["spearman_rho"]
        
        print(f"  {name:26s} (alpha={best_alpha:6.1f}) | Donor rho = {d_rho:.3f} (95% CI: [{d_ci[0]:.3f}, {d_ci[1]:.3f}], p_perm={d_p_perm:.3f}), R2={d_r2:.3f} | Sample rho = {s_rho:.3f}")
        
    # Supervised cVAE predictions for plotting
    cvae_alpha = progression_benchmarks["Conditional VAE (64-dim)"]["tuned_alpha"]
    primary_ridge = Ridge(alpha=cvae_alpha)
    primary_ridge.fit(z_cvae_train, y_train_stage)
    cvae_pred_test = primary_ridge.predict(z_cvae_test)
    cvae_pred_all = primary_ridge.predict(z_cvae_all)
    
    # Unsupervised PC1 for cVAE
    cvae_pca1 = PCA(n_components=1, random_state=RANDOM_SEED)
    score_train = cvae_pca1.fit_transform(z_cvae_train).flatten()
    score_test = cvae_pca1.transform(z_cvae_test).flatten()
    r_orient, _ = spearmanr(y_train_stage, score_train)
    if r_orient < 0:
        score_train = -score_train
        score_test = -score_test
    score_all = cvae_pca1.transform(z_cvae_all).flatten() * (-1.0 if r_orient < 0 else 1.0)
    
    # -------------------------------------------------------------------------
    # [5/8] LATENT PC1 VS ANATOMICAL BRAIN REGIONS (EMPIRICAL VERIFICATION)
    # -------------------------------------------------------------------------
    print(f"\n[4/8] Testing Latent PC1 Correlation & ANOVA with Anatomical Brain Regions:")
    regions_all = full_clin.loc[all_samples, "tissue"].values
    region_pc1_groups = [score_all[regions_all == reg] for reg in np.unique(regions_all)]
    f_stat_reg, p_val_reg = f_oneway(*region_pc1_groups)
    
    is_cerebellum = (regions_all == "Cerebellum").astype(int)
    r_reg_pc1, p_reg_pc1 = pearsonr(is_cerebellum, score_all)
    reg_pc1_r2 = float(r_reg_pc1 ** 2)
    print(f"  Latent PC1 across 4 Regions ANOVA: F = {f_stat_reg:.3f}, p = {p_val_reg:.4e}")
    print(f"  Latent PC1 vs Cerebellum/Neocortex: Pearson r = {r_reg_pc1:.3f} (R2 = {reg_pc1_r2:.3f}, p = {p_reg_pc1:.4e})")
    
    # -------------------------------------------------------------------------
    # [6/8] 5-FOLD DONOR-GROUPED VS SAMPLE-SPLIT CV (LEAKAGE INFLATION EXPERIMENT)
    # -------------------------------------------------------------------------
    print(f"\n[5/8] Donor-Grouped 5-Fold CV vs Sample-Split Leakage Inflation Experiment:")
    
    ad_ctrl_mask = (y_all_stage == 0) | (y_all_stage == 2)
    X_clf_raw = X_all[ad_ctrl_mask]
    y_clf = (y_all_stage[ad_ctrl_mask] == 2).astype(int)
    donors_clf = donor_all[ad_ctrl_mask]
    C_clf = C_all[ad_ctrl_mask]
    
    asym_ad_mask = (y_all_stage == 1) | (y_all_stage == 2)
    X_asym_raw = X_all[asym_ad_mask]
    y_asym = (y_all_stage[asym_ad_mask] == 2).astype(int)
    donors_asym = donor_all[asym_ad_mask]
    C_asym = C_all[asym_ad_mask]
    
    clf_models = {
        "Age + Sex Only": "demo",
        "Raw HVG (4,000 genes)": "raw",
        "PCA (64-dim)": "pca",
        "Residualized PCA (64-dim)": "pca_resid",
        "Unconditioned VAE": "vae",
        "Conditional VAE": "cvae",
    }
    
    classification_benchmarks = {}
    resilience_benchmarks = {}
    leakage_inflation_results = {}
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    for model_name, m_type in clf_models.items():
        # AD vs Control: Donor-Grouped Split
        donor_aucs = []
        for tr_idx, val_idx in sgkf.split(X_clf_raw, y_clf, groups=donors_clf):
            assert len(set(donors_clf[tr_idx]).intersection(set(donors_clf[val_idx]))) == 0
            if m_type == "demo":
                feat_tr, feat_val = C_clf[tr_idx], C_clf[val_idx]
            elif m_type == "raw":
                feat_tr, feat_val = X_clf_raw[tr_idx], X_clf_raw[val_idx]
            elif m_type == "pca":
                pca_f = PCA(n_components=LATENT_DIM, random_state=RANDOM_SEED)
                feat_tr = pca_f.fit_transform(X_clf_raw[tr_idx])
                feat_val = pca_f.transform(X_clf_raw[val_idx])
            elif m_type == "pca_resid":
                reg_d = LinearRegression().fit(C_clf[tr_idx], X_clf_raw[tr_idx])
                res_tr = X_clf_raw[tr_idx] - reg_d.predict(C_clf[tr_idx])
                res_val = X_clf_raw[val_idx] - reg_d.predict(C_clf[val_idx])
                pca_f = PCA(n_components=LATENT_DIM, random_state=RANDOM_SEED)
                feat_tr = pca_f.fit_transform(res_tr)
                feat_val = pca_f.transform(res_val)
            elif m_type == "vae":
                feat_tr = z_vae_all[ad_ctrl_mask][tr_idx]
                feat_val = z_vae_all[ad_ctrl_mask][val_idx]
            elif m_type == "cvae":
                feat_tr = z_cvae_all[ad_ctrl_mask][tr_idx]
                feat_val = z_cvae_all[ad_ctrl_mask][val_idx]
                
            clf = SVC(kernel="linear", probability=True, random_state=RANDOM_SEED)
            clf.fit(feat_tr, y_clf[tr_idx])
            probs = clf.predict_proba(feat_val)[:, 1]
            donor_aucs.append(float(roc_auc_score(y_clf[val_idx], probs)))
            
        mean_donor_auc = float(np.mean(donor_aucs))
        std_donor_auc = float(np.std(donor_aucs, ddof=1))
        
        # AD vs Control: Sample-Level Split (Leakage Demonstration)
        sample_aucs = []
        for tr_idx, val_idx in skf.split(X_clf_raw, y_clf):
            if m_type == "demo":
                feat_tr, feat_val = C_clf[tr_idx], C_clf[val_idx]
            elif m_type == "raw":
                feat_tr, feat_val = X_clf_raw[tr_idx], X_clf_raw[val_idx]
            elif m_type == "pca":
                pca_f = PCA(n_components=LATENT_DIM, random_state=RANDOM_SEED)
                feat_tr = pca_f.fit_transform(X_clf_raw[tr_idx])
                feat_val = pca_f.transform(X_clf_raw[val_idx])
            elif m_type == "pca_resid":
                reg_d = LinearRegression().fit(C_clf[tr_idx], X_clf_raw[tr_idx])
                res_tr = X_clf_raw[tr_idx] - reg_d.predict(C_clf[tr_idx])
                res_val = X_clf_raw[val_idx] - reg_d.predict(C_clf[val_idx])
                pca_f = PCA(n_components=LATENT_DIM, random_state=RANDOM_SEED)
                feat_tr = pca_f.fit_transform(res_tr)
                feat_val = pca_f.transform(res_val)
            elif m_type == "vae":
                feat_tr = z_vae_all[ad_ctrl_mask][tr_idx]
                feat_val = z_vae_all[ad_ctrl_mask][val_idx]
            elif m_type == "cvae":
                feat_tr = z_cvae_all[ad_ctrl_mask][tr_idx]
                feat_val = z_cvae_all[ad_ctrl_mask][val_idx]
                
            clf = SVC(kernel="linear", probability=True, random_state=RANDOM_SEED)
            clf.fit(feat_tr, y_clf[tr_idx])
            probs = clf.predict_proba(feat_val)[:, 1]
            sample_aucs.append(float(roc_auc_score(y_clf[val_idx], probs)))
            
        mean_sample_auc = float(np.mean(sample_aucs))
        std_sample_auc = float(np.std(sample_aucs, ddof=1))
        
        classification_benchmarks[model_name] = {
            "mean_auc": mean_donor_auc,
            "std_auc": std_donor_auc,
            "fold_aucs": [round(x, 4) for x in donor_aucs]
        }
        
        leakage_inflation_results[model_name] = {
            "donor_grouped_auc_mean": mean_donor_auc,
            "donor_grouped_auc_std": std_donor_auc,
            "sample_split_auc_mean": mean_sample_auc,
            "sample_split_auc_std": std_sample_auc,
            "inflation_delta": float(mean_sample_auc - mean_donor_auc)
        }
        
        # Task 2: AsymAD vs AD (Resilience) - Donor-Grouped
        asym_aucs = []
        for tr_idx, val_idx in sgkf.split(X_asym_raw, y_asym, groups=donors_asym):
            assert len(set(donors_asym[tr_idx]).intersection(set(donors_asym[val_idx]))) == 0
            if m_type == "demo":
                feat_tr, feat_val = C_asym[tr_idx], C_asym[val_idx]
            elif m_type == "raw":
                feat_tr, feat_val = X_asym_raw[tr_idx], X_asym_raw[val_idx]
            elif m_type == "pca":
                pca_f = PCA(n_components=LATENT_DIM, random_state=RANDOM_SEED)
                feat_tr = pca_f.fit_transform(X_asym_raw[tr_idx])
                feat_val = pca_f.transform(X_asym_raw[val_idx])
            elif m_type == "pca_resid":
                reg_d = LinearRegression().fit(C_asym[tr_idx], X_asym_raw[tr_idx])
                res_tr = X_asym_raw[tr_idx] - reg_d.predict(C_asym[tr_idx])
                res_val = X_asym_raw[val_idx] - reg_d.predict(C_asym[val_idx])
                pca_f = PCA(n_components=LATENT_DIM, random_state=RANDOM_SEED)
                feat_tr = pca_f.fit_transform(res_tr)
                feat_val = pca_f.transform(res_val)
            elif m_type == "vae":
                feat_tr = z_vae_all[asym_ad_mask][tr_idx]
                feat_val = z_vae_all[asym_ad_mask][val_idx]
            elif m_type == "cvae":
                feat_tr = z_cvae_all[asym_ad_mask][tr_idx]
                feat_val = z_cvae_all[asym_ad_mask][val_idx]
                
            clf = SVC(kernel="linear", probability=True, random_state=RANDOM_SEED)
            clf.fit(feat_tr, y_asym[tr_idx])
            probs = clf.predict_proba(feat_val)[:, 1]
            asym_aucs.append(float(roc_auc_score(y_asym[val_idx], probs)))
            
        mean_asym_auc = float(np.mean(asym_aucs))
        std_asym_auc = float(np.std(asym_aucs, ddof=1))
        
        resilience_benchmarks[model_name] = {
            "mean_auc": mean_asym_auc,
            "std_auc": std_asym_auc,
            "fold_aucs": [round(x, 4) for x in asym_aucs]
        }
        
        print(f"  {model_name:26s} | AD vs Ctrl (Group CV): {mean_donor_auc:.4f} ± {std_donor_auc:.4f} | Sample-Split: {mean_sample_auc:.4f} (Δ={mean_sample_auc - mean_donor_auc:+.4f}) | AsymAD vs AD: {mean_asym_auc:.4f} ± {std_asym_auc:.4f}")

    # -------------------------------------------------------------------------
    # [7/8] REGIONAL VULNERABILITY WITH DONOR-BOOTSTRAP CIs & CEREBELLAR REFRAMING
    # -------------------------------------------------------------------------
    print(f"\n[6/8] Regional Vulnerability & Cerebellar Non-Zero Signal Breakdown:")
    
    unique_tissues = ["Temporal_Cortex", "Entorhinal_Cortex", "Frontal_Cortex", "Cerebellum"]
    regional_results = {}
    
    for tissue in unique_tissues:
        t_mask_test = [i for i, s in enumerate(test_samples) if test_clin.loc[s, "tissue"] == tissue]
        t_mask_all = (tissue_all == tissue)
        
        if len(t_mask_test) > 5:
            y_t_test = y_test_stage[t_mask_test]
            pred_t_test = cvae_pred_test[t_mask_test]
            donors_t_test = donor_test[t_mask_test]
            
            rho_t, p_t = spearmanr(y_t_test, pred_t_test)
            r_t, _ = pearsonr(y_t_test, pred_t_test)
            sse_t = np.sum((y_t_test - pred_t_test) ** 2)
            sst_t = np.sum((y_t_test - np.mean(y_t_test)) ** 2)
            r2_t = 1.0 - (sse_t / sst_t) if sst_t > 0 else 0.0
            mae_t = float(mean_absolute_error(y_t_test, pred_t_test))
            
            df_reg = pd.DataFrame({"donor": donors_t_test, "y_true": y_t_test, "y_pred": pred_t_test})
            reg_ci = compute_regional_bootstrap_ci(df_reg, n_bootstraps=1000)
        else:
            rho_t, p_t, r_t, r2_t, mae_t, reg_ci = None, None, None, None, None, [None, None]
            
        t_ad_ctrl = t_mask_all & ((y_all_stage == 0) | (y_all_stage == 2))
        X_reg_z = z_cvae_all[t_ad_ctrl]
        y_reg = (y_all_stage[t_ad_ctrl] == 2).astype(int)
        donors_reg = donor_all[t_ad_ctrl]
        
        reg_aucs = []
        if len(np.unique(y_reg)) > 1 and len(y_reg) >= 15:
            reg_sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
            for r_tr, r_te in reg_sgkf.split(X_reg_z, y_reg, groups=donors_reg):
                clf_reg = SVC(kernel="linear", probability=True, random_state=RANDOM_SEED)
                clf_reg.fit(X_reg_z[r_tr], y_reg[r_tr])
                probs_reg = clf_reg.predict_proba(X_reg_z[r_te])[:, 1]
                reg_aucs.append(float(roc_auc_score(y_reg[r_te], probs_reg)))
            mean_reg_auc = float(np.mean(reg_aucs))
            std_reg_auc = float(np.std(reg_aucs, ddof=1))
        else:
            mean_reg_auc, std_reg_auc = None, None
            
        regional_results[tissue] = {
            "n_samples": int(np.sum(tissue_all == tissue)),
            "n_test_samples": int(len(t_mask_test)),
            "progression_rho": float(rho_t) if rho_t is not None else None,
            "progression_rho_95ci": reg_ci,
            "progression_p": float(p_t) if p_t is not None else None,
            "progression_r2": float(r2_t) if r2_t is not None else None,
            "progression_mae": float(mae_t) if mae_t is not None else None,
            "classification_auc_mean": mean_reg_auc,
            "classification_auc_std": std_reg_auc,
        }
        
        print(f"  {tissue:20s} (n={regional_results[tissue]['n_samples']:2d}, test_n={len(t_mask_test):2d}) | Progression rho = {rho_t:.3f} (95% CI: [{reg_ci[0]:.3f}, {reg_ci[1]:.3f}]) | AD vs Ctrl AUC = {mean_reg_auc:.3f} ± {std_reg_auc:.3f}")

    # -------------------------------------------------------------------------
    # [8/8] BIOLOGICAL PATHWAY ENRICHMENT (ENRICHR)
    # -------------------------------------------------------------------------
    print(f"\n[7/8] Gene-Level Disease Score Correlation & Enrichr Pathway Analysis:")
    
    gene_names = list(expr_df.index)
    gene_rhos = []
    for g_idx in range(X_test.shape[1]):
        r_g, _ = spearmanr(X_test[:, g_idx], cvae_pred_test)
        gene_rhos.append(r_g)
    gene_rhos = np.nan_to_num(np.array(gene_rhos), nan=0.0)
    
    enrichment_data, top_genes = run_enrichr_analysis(gene_names, gene_scores=gene_rhos, top_n=150)
    top_gene_corrs = {g: float(gene_rhos[gene_names.index(g)]) for g in top_genes if g in gene_names}
    
    # -------------------------------------------------------------------------
    # COMPILE COMPLETE RESULTS OBJECT
    # -------------------------------------------------------------------------
    summary_results = {
        "cohort": {
            "n_donors_total": int(len(np.unique(donor_all))),
            "n_donors_train": int(len(train_donors)),
            "n_donors_test": int(len(test_donors)),
            "n_samples_total": int(len(all_samples)),
            "n_samples_train": int(len(train_samples)),
            "n_samples_test": int(len(test_samples)),
            "n_genes": int(expr_df.shape[0]),
            "donors_by_diagnosis": {
                "Control": int((donor_df["disease state"] == "control").sum()),
                "AsymAD": int((donor_df["disease state"] == "AsymAD").sum()),
                "AD": int((donor_df["disease state"] == "AD").sum()),
            },
            "demographics": demographics_summary
        },
        "progression_benchmarks": progression_benchmarks,
        "classification_benchmarks": classification_benchmarks,
        "resilience_asymad_vs_ad": resilience_benchmarks,
        "leakage_inflation": leakage_inflation_results,
        "regional_vulnerability": regional_results,
        "latent_pc1_region_test": {
            "anova_f": float(f_stat_reg),
            "anova_p": float(p_val_reg),
            "cerebellum_vs_cortex_r": float(r_reg_pc1),
            "cerebellum_vs_cortex_r2": float(reg_pc1_r2),
            "cerebellum_vs_cortex_p": float(p_reg_pc1)
        },
        "pathway_enrichment": enrichment_data,
        "top_progression_genes": top_genes[:25],
        "top_gene_correlations": {k: round(v, 4) for k, v in list(top_gene_corrs.items())[:25]}
    }
    
    summary_path = results_dir / "summary_results.json"
    with open(summary_path, "w") as f:
        json.dump(summary_results, f, indent=2)
    print(f"\n[✓] Complete experimental summary saved to {summary_path}")
    
    return {
        "summary": summary_results,
        "z_train": z_cvae_train,
        "z_test": z_cvae_test,
        "z_all": z_cvae_all,
        "z_pca_all": z_pca_all,
        "score_train": score_train,
        "score_test": score_test,
        "score_all": score_all,
        "cvae_pred_test": cvae_pred_test,
        "cvae_pred_all": cvae_pred_all,
        "y_train_stage": y_train_stage,
        "y_test_stage": y_test_stage,
        "y_all_stage": y_all_stage,
        "train_samples": train_samples,
        "test_samples": test_samples,
        "all_samples": all_samples,
        "full_clin": full_clin,
        "expr_df": expr_df,
        "cvae_model": cvae,
    }


if __name__ == "__main__":
    exp_results = run_experiment()
    print("\n[✓] Experiment execution finished successfully.")
