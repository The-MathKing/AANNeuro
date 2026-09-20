"""
Covariate-Residualized Principal Component Analysis (In-Fold).
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


class InFoldResidualizedPCA:
    """
    Fits Covariate Residualization + PCA strictly on training fold data.
    """
    def __init__(self, n_components=10, top_hvg=2000):
        self.n_components = n_components
        self.top_hvg = top_hvg
        self.scaler = None
        self.pca = None
        self.selected_genes = None
        self.cov_beta = None

    def fit(self, X_train, cov_train=None):
        """
        X_train: DataFrame (samples x genes)
        cov_train: DataFrame or array (samples x covariates)
        """
        if self.top_hvg and self.top_hvg < X_train.shape[1]:
            variances = X_train.var(axis=0)
            self.selected_genes = variances.nlargest(self.top_hvg).index.tolist()
            X_sub = X_train[self.selected_genes].values
        else:
            self.selected_genes = list(X_train.columns)
            X_sub = X_train.values

        # Covariate residualization
        if cov_train is not None:
            cov_mat = sm.add_constant(np.asarray(cov_train), has_constant='add')
            self.cov_beta = np.linalg.lstsq(cov_mat, X_sub, rcond=None)[0]
            X_resid = X_sub - cov_mat @ self.cov_beta
        else:
            self.cov_beta = None
            X_resid = X_sub

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_resid)

        self.pca = PCA(n_components=min(self.n_components, X_scaled.shape[0] - 1, X_scaled.shape[1]))
        self.pca.fit(X_scaled)
        return self

    def transform(self, X, cov=None):
        """Transform test data."""
        X_sub = X[self.selected_genes].values
        if self.cov_beta is not None and cov is not None:
            cov_mat = sm.add_constant(np.asarray(cov), has_constant='add')
            X_resid = X_sub - cov_mat @ self.cov_beta
        else:
            X_resid = X_sub

        X_scaled = self.scaler.transform(X_resid)
        return self.pca.transform(X_scaled)
