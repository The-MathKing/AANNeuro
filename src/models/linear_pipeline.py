"""
Leakage-Aware In-Fold Linear Pipelines (Ridge, ElasticNet, Logistic Regression).
All feature selection, z-scoring, and covariate residualization are fit STRICTLY inside training folds.
Supports statistical HVG selection, biological prior filtering, and hybrid prior-guided architectures.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LogisticRegressionCV, ElasticNetCV, Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# Canonical Biological Prior Panels (AMP-AD Consensus Modules & Validated GWAS Risk Loci)
CANONICAL_BIOLOGICAL_PRIORS = [
    # Synaptic / Neuronal
    "SYN1", "SYN2", "SYP", "SNAP25", "VAMP2", "STX1A", "BSN", "PCLO", "SLC17A7", "CAMK2A", "GRIN1", "GRIN2A", "GRIA1", "DLG4", "NEFL",
    # Microglial / Immune
    "TYROBP", "TREM2", "AIF1", "CD68", "C1QA", "C1QB", "C1QC", "FCER1G", "ITGAM", "MS4A6A", "HLA-DRA", "CX3CR1", "SPI1", "TLR2", "LAPTM5",
    # ECM / Vascular
    "COL1A1", "COL1A2", "COL3A1", "COL4A1", "COL4A2", "FN1", "VWF", "PECAM1", "CLDN5", "CD34", "CDH5", "LAMB1", "LAMA4", "SPARC", "BGN",
    # Mitochondrial / Metabolic
    "NDUFA1", "NDUFB1", "NDUFS1", "SDHA", "SDHB", "UQCRC1", "UQCRC2", "COX4I1", "COX5A", "ATP5F1A", "ATP5F1B", "SOD2", "PRDX3", "VDAC1",
    # AD GWAS Risk Loci
    "APOE", "BIN1", "CLU", "ABCA7", "CR1", "PICALM", "CD33", "MS4A4A", "CD2AP", "EPHA1", "PTK2B", "CASS4", "INPP5D", "MEF2C",
    "SORL1", "ADAM10", "PLCG2", "ABI3", "APH1B", "APP", "PSEN1", "PSEN2", "MAPT"
]


def residualize_covariates(X_train, cov_train, X_test=None, cov_test=None):
    """
    Regress out confounding covariates (e.g. age, sex, PMI) using OLS fit strictly on training set.
    X_train: (n_train, n_features)
    cov_train: (n_train, n_covariates)
    """
    cov_train_mat = sm.add_constant(cov_train, has_constant='add')
    try:
        beta = np.linalg.lstsq(cov_train_mat, X_train, rcond=None)[0]
        X_train_resid = X_train - cov_train_mat @ beta
        
        if X_test is not None and cov_test is not None:
            cov_test_mat = sm.add_constant(cov_test, has_constant='add')
            X_test_resid = X_test - cov_test_mat @ beta
            return X_train_resid, X_test_resid
        return X_train_resid
    except Exception:
        if X_test is not None:
            return X_train, X_test
        return X_train


class InFoldLinearModel:
    """
    Standardized in-fold linear classifier / regressor with support for
    HVG/Prior feature selection, within-sample rank or z-score transforms, and L2/L1/SVM classifiers.
    """
    def __init__(self, model_type="ridge_classifier", alphas=(0.1, 1.0, 10.0, 100.0), top_hvg=2000, 
                 feature_strategy="hvg", transform_strategy="zscore", prior_genes=None, seed=42):
        self.model_type = model_type
        self.alphas = alphas
        self.top_hvg = top_hvg
        self.feature_strategy = feature_strategy
        self.transform_strategy = transform_strategy
        self.prior_genes = prior_genes if prior_genes is not None else CANONICAL_BIOLOGICAL_PRIORS
        self.seed = seed
        self.scaler = None
        self.selected_genes = None
        self.model = None
        self.constant_pred = None

    def fit_transform_train(self, X_train, y_train):
        """
        Fit feature selection, scaler, and model strictly on training data.
        X_train: DataFrame (samples x genes)
        y_train: Series or 1D array
        """
        # 1. Feature selection fit strictly in-fold
        if self.feature_strategy == "biological_prior":
            avail_priors = [g for g in self.prior_genes if g in X_train.columns]
            self.selected_genes = avail_priors if avail_priors else list(X_train.columns)
        elif self.feature_strategy == "hybrid_prior":
            avail_priors = [g for g in self.prior_genes if g in X_train.columns]
            k_hvg = self.top_hvg if self.top_hvg and self.top_hvg < X_train.shape[1] else 500
            variances = X_train.var(axis=0)
            top_hvg = variances.nlargest(k_hvg).index.tolist()
            self.selected_genes = list(dict.fromkeys(top_hvg + avail_priors))
        else: # Default "hvg"
            if self.top_hvg and self.top_hvg < X_train.shape[1]:
                variances = X_train.var(axis=0)
                self.selected_genes = variances.nlargest(self.top_hvg).index.tolist()
            else:
                self.selected_genes = list(X_train.columns)

        X_sub = X_train[self.selected_genes].copy()

        # 2. Within-sample relative transformation (if configured)
        if self.transform_strategy == "rank":
            X_sub = X_sub.rank(axis=1, pct=True)

        # 3. Scaler fit on train only
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_sub.values)
        self.constant_pred = None

        # 4. Model fit
        if self.model_type in ["ridge_classifier", "logistic_l1", "logistic_l2", "linear_svm"]:
            unique_classes, counts = np.unique(y_train, return_counts=True)
            if len(unique_classes) < 2:
                self.constant_pred = float(unique_classes[0])
                self.model = None
                return self

            min_count = int(np.min(counts))
            inner_cv = max(2, min(3, min_count))

            if self.model_type == "ridge_classifier":
                self.model = LogisticRegressionCV(
                    Cs=5, cv=inner_cv, scoring='roc_auc', solver='lbfgs', max_iter=1000, random_state=self.seed
                )
                try:
                    self.model.fit(X_scaled, y_train)
                except Exception:
                    self.model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=self.seed)
                    self.model.fit(X_scaled, y_train)
            elif self.model_type == "logistic_l2":
                self.model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=self.seed)
                self.model.fit(X_scaled, y_train)
            elif self.model_type == "linear_svm":
                from sklearn.svm import SVC
                self.model = SVC(kernel='linear', C=0.05, probability=True, random_state=self.seed)
                self.model.fit(X_scaled, y_train)
            else:
                self.model = LogisticRegression(C=1.0, penalty='l1', solver='liblinear', max_iter=1000, random_state=self.seed)
                self.model.fit(X_scaled, y_train)

        elif self.model_type == "ridge_regression":
            self.model = RidgeCV(alphas=self.alphas, cv=min(3, len(X_train) - 1))
            self.model.fit(X_scaled, y_train)
        elif self.model_type == "elastic_net":
            self.model = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9], cv=min(3, len(X_train) - 1), random_state=self.seed, max_iter=2000)
            self.model.fit(X_scaled, y_train)
        else:
            self.model = Ridge(alpha=1.0)
            self.model.fit(X_scaled, y_train)

        return self

    def predict(self, X_test):
        """Predict on test data using in-fold fitted transformations."""
        if isinstance(X_test, pd.DataFrame):
            X_sub = X_test.reindex(columns=self.selected_genes, fill_value=0.0).copy()
        else:
            X_sub = pd.DataFrame(np.asarray(X_test), columns=self.selected_genes)

        if self.constant_pred is not None:
            return np.full(len(X_sub), self.constant_pred)

        if self.transform_strategy == "rank":
            X_sub = X_sub.rank(axis=1, pct=True)

        X_scaled = self.scaler.transform(X_sub.values)
        
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_scaled)
            if probs.shape[1] == 2:
                return probs[:, 1]
            return probs
        elif hasattr(self.model, "decision_function"):
            return self.model.decision_function(X_scaled)
        else:
            return self.model.predict(X_scaled)

    def get_feature_importances(self):
        """Extract signed coefficients for selected genes."""
        if self.model is None or not hasattr(self.model, "coef_"):
            return pd.Series(0.0, index=self.selected_genes)
        coefs = self.model.coef_.flatten()
        return pd.Series(coefs, index=self.selected_genes)
