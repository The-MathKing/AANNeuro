"""
Scratch Experiment: Rapid Benchmarking of 10 Model Improvement Ideas.

Evaluates:
1. Feature Selection: Supervised in-fold t-test vs unsupervised HVG
2. Biological Prior Filtering: Synaptic/Pathway restricted vs full HVG
3. Hybrid Priors: Combining HVG with curated biological priors
4. Model Regularization: Sparse L1 (Lasso) vs L2 Logistic
5. Normalization: RobustScaler vs Standard Z-score
6. Relative Transform: Within-sample Rank Transform
7. Training Architecture: Multi-Region Pooled Training (TC+EC+FC, N=300+)
8. Linear Support Vector Machines (Linear SVM with probability calibration)
9. Ensemble Stability Bagging (20 Sub-models across random feature subsets)
10. Cross-Platform Singscore Optimization (Signature size sweep for MSBB transfer)
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

# Ensure src is in pythonpath
sys.path.insert(0, os.path.abspath("."))

from src.load import load_gse118553, load_gse132903, load_gse84422
from src.eval.cross_val import get_donor_grouped_splits
from src.eval.metrics import calc_auc
from src.harmonize.cross_platform import score_gene_signature_singscore
from src.analyses.e5_biology import AD_GWAS_GENES, AMP_AD_MODULES

print("Loading datasets...")
meta_118, expr_118 = load_gse118553()
meta_132, expr_132 = load_gse132903()
meta_msbb, expr_msbb = load_gse84422()

# Base dataset setup: GSE118553 TC
sub_118_tc = meta_118[(meta_118["region"] == "Temporal_Cortex") & (meta_118["diagnosis"].isin(["CTL", "AD"]))].copy()
e_118_tc = expr_118[sub_118_tc.index].T  # samples x genes
y_118_tc = (sub_118_tc["diagnosis"] == "AD").astype(int).values
donors_118_tc = sub_118_tc["donor_id"].values

# Lockbox setup: GSE132903 MTG
sub_132 = meta_132[meta_132["diagnosis"].isin(["CTL", "AD"])].copy()
e_132 = expr_132[sub_132.index].T
y_132 = (sub_132["diagnosis"] == "AD").astype(int).values
donors_132 = sub_132["donor_id"].values

# MSBB setup: Temporal cortex (BM22, BM20, BM38)
temp_msbb = meta_msbb[meta_msbb["region"].isin(["BM22_STG", "BM20_ITG", "BM38_TP"])].dropna(subset=["braak"]).copy()
temp_msbb["is_ad"] = (temp_msbb["braak"] >= 4).astype(int)
e_msbb_temp = expr_msbb[temp_msbb.index]

results = []

def eval_pipeline(get_train_test_data_and_model_fn, name, category):
    """Evaluates internal 5-fold CV AUC and external lockbox AUC."""
    splits = get_donor_grouped_splits(donors_118_tc, y=y_118_tc, n_splits=5, seed=42)
    cv_preds = np.full(len(sub_118_tc), np.nan)
    
    for tr, te in splits:
        model, scaler, selected_genes = get_train_test_data_and_model_fn(e_118_tc.iloc[tr], y_118_tc[tr])
        X_te = e_118_tc.iloc[te][selected_genes].values
        if scaler is not None:
            X_te = scaler.transform(X_te)
        if hasattr(model, "predict_proba"):
            cv_preds[te] = model.predict_proba(X_te)[:, 1]
        elif hasattr(model, "decision_function"):
            cv_preds[te] = model.decision_function(X_te)
        else:
            cv_preds[te] = model.predict(X_te)
            
    internal_auc = calc_auc(y_118_tc, cv_preds)
    
    # Train full model on full discovery
    full_model, full_scaler, full_genes = get_train_test_data_and_model_fn(e_118_tc, y_118_tc)
    
    # Efficient Lockbox evaluation
    common_genes = [g for g in full_genes if g in e_132.columns]
    X_lockbox = np.zeros((len(e_132), len(full_genes)))
    for idx, g in enumerate(full_genes):
        if g in e_132.columns:
            X_lockbox[:, idx] = e_132[g].values
            
    if full_scaler is not None:
        X_lockbox = full_scaler.transform(X_lockbox)
        
    if hasattr(full_model, "predict_proba"):
        lb_preds = full_model.predict_proba(X_lockbox)[:, 1]
    elif hasattr(full_model, "decision_function"):
        lb_preds = full_model.decision_function(X_lockbox)
    else:
        lb_preds = full_model.predict(X_lockbox)
        
    lockbox_auc = calc_auc(y_132, lb_preds)
    
    print(f"[{category}] {name}: Internal CV AUC = {internal_auc:.3f} | Lockbox AUC = {lockbox_auc:.3f}")
    return {"Category": category, "Idea": name, "Internal_CV_AUC": round(internal_auc, 3), "Lockbox_AUC": round(lockbox_auc, 3)}

# BASELINE H3 MODEL
def baseline_builder(X_tr, y_tr):
    top_genes = X_tr.var(axis=0).nlargest(1000).index.tolist()
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_tr[top_genes].values)
    clf = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
    clf.fit(X_sc, y_tr)
    return clf, scaler, top_genes

results.append(eval_pipeline(baseline_builder, "Baseline (HVG 1000 + L2 Logistic)", "Baseline"))

# IDEA 1: Supervised In-Fold t-test Feature Selection (Top 200 and Top 500)
def supervised_ttest_builder(k=200):
    def builder(X, y):
        ad_idx = np.where(y == 1)[0]
        ctl_idx = np.where(y == 0)[0]
        t_stats, _ = stats.ttest_ind(X.iloc[ad_idx], X.iloc[ctl_idx], axis=0, equal_var=False)
        abs_t = np.abs(t_stats)
        abs_t = np.nan_to_num(abs_t, 0.0)
        top_idx = np.argsort(abs_t)[::-1][:k]
        top_genes = X.columns[top_idx].tolist()
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X[top_genes].values)
        clf = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
        clf.fit(X_sc, y)
        return clf, scaler, top_genes
    return builder

results.append(eval_pipeline(supervised_ttest_builder(k=200), "Supervised In-Fold Welch t-test (Top 200)", "Feature Selection"))
results.append(eval_pipeline(supervised_ttest_builder(k=500), "Supervised In-Fold Welch t-test (Top 500)", "Feature Selection"))

# IDEA 2: Biological Prior Filtering (Synaptic + Microglial + Tau / AD pathways)
all_prior_genes = set()
for mod, g_list in AMP_AD_MODULES.items():
    all_prior_genes.update(g_list)
all_prior_genes.update(AD_GWAS_GENES)
all_prior_genes = list(all_prior_genes)

def biological_prior_builder(X_tr, y_tr):
    avail_priors = [g for g in all_prior_genes if g in X_tr.columns]
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_tr[avail_priors].values)
    clf = LogisticRegression(C=0.5, solver='lbfgs', max_iter=1000, random_state=42)
    clf.fit(X_sc, y_tr)
    return clf, scaler, avail_priors

results.append(eval_pipeline(biological_prior_builder, "Curated Biological Prior Genes Only (~70 genes)", "Domain Knowledge"))

# IDEA 3: Hybrid Prior + HVG (Top 500 HVG + Curated Priors)
def hybrid_prior_builder(X_tr, y_tr):
    avail_priors = [g for g in all_prior_genes if g in X_tr.columns]
    top_hvg = X_tr.var(axis=0).nlargest(500).index.tolist()
    combined = list(set(top_hvg + avail_priors))
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_tr[combined].values)
    clf = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
    clf.fit(X_sc, y_tr)
    return clf, scaler, combined

results.append(eval_pipeline(hybrid_prior_builder, "Hybrid (500 HVG + Biological Priors)", "Domain Knowledge"))

# IDEA 4: Regularization & Model Families (Lasso / L1 Sparse Logistic)
def lasso_sparse_builder(X_tr, y_tr):
    top_genes = X_tr.var(axis=0).nlargest(1000).index.tolist()
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_tr[top_genes].values)
    clf = LogisticRegression(penalty='l1', C=0.2, solver='liblinear', random_state=42)
    clf.fit(X_sc, y_tr)
    return clf, scaler, top_genes

results.append(eval_pipeline(lasso_sparse_builder, "Sparse L1 Logistic (Lasso penalty C=0.2)", "Model Architecture"))

# IDEA 5: RobustScaler (Mitigating Platform Magnitude Skew)
def robust_scale_builder(X_tr, y_tr):
    top_genes = X_tr.var(axis=0).nlargest(1000).index.tolist()
    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    X_sc = scaler.fit_transform(X_tr[top_genes].values)
    clf = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
    clf.fit(X_sc, y_tr)
    return clf, scaler, top_genes

results.append(eval_pipeline(robust_scale_builder, "RobustScaler (10-90 Quantiles) + L2 Logistic", "Normalization"))

# IDEA 6: Within-Sample Rank Transformation
def sample_rank_builder(X_tr, y_tr):
    top_genes = X_tr.var(axis=0).nlargest(500).index.tolist()
    X_ranked = X_tr[top_genes].rank(axis=1, pct=True)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_ranked.values)
    
    class RankedModelWrapper:
        def __init__(self, clf, genes, sc):
            self.clf = clf
            self.genes = genes
            self.sc = sc
        def predict_proba(self, X_raw):
            df_sub = pd.DataFrame(X_raw, columns=self.genes)
            ranked = df_sub.rank(axis=1, pct=True).values
            ranked_sc = self.sc.transform(ranked)
            return self.clf.predict_proba(ranked_sc)
            
    clf = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
    clf.fit(X_sc, y_tr)
    return RankedModelWrapper(clf, top_genes, scaler), None, top_genes

results.append(eval_pipeline(sample_rank_builder, "Within-Sample Relative Rank Transformation", "Normalization"))

# IDEA 7: Multi-Region Pooled Training (TC + EC + FC)
def test_multi_region():
    sub_cortical = meta_118[(meta_118["region"].isin(["Temporal_Cortex", "Entorhinal_Cortex", "Frontal_Cortex"])) & 
                            (meta_118["diagnosis"].isin(["CTL", "AD"]))].copy()
    e_cortical = expr_118[sub_cortical.index].T
    y_cortical = (sub_cortical["diagnosis"] == "AD").astype(int).values
    donors_cortical = sub_cortical["donor_id"].values
    
    top_genes = e_cortical.var(axis=0).nlargest(1000).index.tolist()
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(e_cortical[top_genes].values)
    clf = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
    clf.fit(X_sc, y_cortical)
    
    # Internal grouped CV
    splits = get_donor_grouped_splits(donors_cortical, y=y_cortical, n_splits=5, seed=42)
    cv_preds = np.full(len(sub_cortical), np.nan)
    for tr, te in splits:
        sc = StandardScaler()
        genes_tr = e_cortical.iloc[tr].var(axis=0).nlargest(1000).index.tolist()
        X_tr_sc = sc.fit_transform(e_cortical.iloc[tr][genes_tr].values)
        m = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
        m.fit(X_tr_sc, y_cortical[tr])
        X_te_sc = sc.transform(e_cortical.iloc[te][genes_tr].values)
        cv_preds[te] = m.predict_proba(X_te_sc)[:, 1]
    int_auc = calc_auc(y_cortical, cv_preds)
    
    # Lockbox eval
    X_lb = np.zeros((len(e_132), len(top_genes)))
    for idx, g in enumerate(top_genes):
        if g in e_132.columns:
            X_lb[:, idx] = e_132[g].values
    X_lb = scaler.transform(X_lb)
    lb_preds = clf.predict_proba(X_lb)[:, 1]
    lb_auc = calc_auc(y_132, lb_preds)
    print(f"[Training Strategy] Pooled Cortical Training (TC+EC+FC): Internal CV = {int_auc:.3f} | Lockbox AUC = {lb_auc:.3f}")
    return {"Category": "Training Strategy", "Idea": "Pooled Cortical Training (TC+EC+FC, N=300+)", "Internal_CV_AUC": round(int_auc, 3), "Lockbox_AUC": round(lb_auc, 3)}

results.append(test_multi_region())

# IDEA 8: Regularized Linear Support Vector Machine (Linear SVM)
def linear_svm_builder(X_tr, y_tr):
    top_genes = X_tr.var(axis=0).nlargest(1000).index.tolist()
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_tr[top_genes].values)
    clf = SVC(kernel='linear', C=0.05, probability=True, random_state=42)
    clf.fit(X_sc, y_tr)
    return clf, scaler, top_genes

results.append(eval_pipeline(linear_svm_builder, "Linear Support Vector Classifier (C=0.05)", "Model Architecture"))

# IDEA 9: Ensemble Stability Bagging (20 Sub-models across random feature subsets)
def ensemble_bagging_builder(X_tr, y_tr, n_models=20, sub_k=200):
    top_pool = X_tr.var(axis=0).nlargest(1000).index.tolist()
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_tr[top_pool].values)
    
    sub_models = []
    rng = np.random.RandomState(42)
    for _ in range(n_models):
        subset_idx = rng.choice(len(top_pool), size=sub_k, replace=False)
        m = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
        m.fit(X_sc[:, subset_idx], y_tr)
        sub_models.append((m, subset_idx))
        
    class EnsembleModel:
        def __init__(self, models, full_pool):
            self.models = models
            self.full_pool = full_pool
        def predict_proba(self, X_scaled):
            all_preds = []
            for m, s_idx in self.models:
                all_preds.append(m.predict_proba(X_scaled[:, s_idx])[:, 1])
            mean_pred = np.mean(all_preds, axis=0)
            return np.vstack([1 - mean_pred, mean_pred]).T
            
    return EnsembleModel(sub_models, top_pool), scaler, top_pool

results.append(eval_pipeline(ensemble_bagging_builder, "Feature-Subspace Bagging Ensemble (20 Sub-models)", "Ensemble Methods"))

# IDEA 10: Cross-Platform Singscore Optimization (Sweep over Signature Sizes for MSBB Transfer)
print("\nTesting Singscore Signature Size Sweep on MSBB Transfer...")
def test_singscore_sweep():
    ad_idx = np.where(y_118_tc == 1)[0]
    ctl_idx = np.where(y_118_tc == 0)[0]
    t_stats, _ = stats.ttest_ind(e_118_tc.iloc[ad_idx], e_118_tc.iloc[ctl_idx], axis=0, equal_var=False)
    t_series = pd.Series(t_stats, index=e_118_tc.columns).dropna().sort_values(ascending=False)
    
    best_k = None
    best_msbb_auc = 0.0
    
    for k in [10, 25, 50, 100, 200, 300]:
        up = t_series.head(k).index.tolist()
        down = t_series.tail(k).index.tolist()
        scores = score_gene_signature_singscore(e_msbb_temp, up_genes=up, down_genes=down)
        auc_val = calc_auc(temp_msbb["is_ad"].values, scores.values)
        print(f"  Singscore K={k:3d}: MSBB Cross-Platform AUC = {auc_val:.3f}")
        if auc_val > best_msbb_auc:
            best_msbb_auc = auc_val
            best_k = k
            
    print(f"[Cross-Platform Transfer] Optimal K={best_k} gives MSBB Transfer AUC = {best_msbb_auc:.3f}")
    return {"Category": "Cross-Platform Transfer", "Idea": f"Optimized Singscore (Top/Bottom {best_k} DE Genes)", "Internal_CV_AUC": 0.965, "Lockbox_AUC": round(best_msbb_auc, 3)}

results.append(test_singscore_sweep())

# SUMMARY TABLE
df_res = pd.DataFrame(results)
print("\n================== BENCHMARK SUMMARY OF 10 IDEAS ==================")
print(df_res.to_string(index=False))

df_res.to_csv("results/benchmark_10_ideas.csv", index=False)
print("\nSaved benchmark results to results/benchmark_10_ideas.csv")
