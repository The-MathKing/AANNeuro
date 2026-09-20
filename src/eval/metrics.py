"""
Evaluation Metrics for Classification and Regression.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score, mean_squared_error
from scipy import stats


def calc_auc(y_true, y_pred_proba):
    """Compute binary ROC-AUC safely."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred_proba)
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_t = y_true[valid]
    y_p = y_pred[valid]
    if len(np.unique(y_t)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y_t, y_p))
    except Exception:
        return np.nan


def calc_spearman(y_true, y_pred):
    """Compute Spearman rank correlation rho and p-value."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_t = y_true[valid]
    y_p = y_pred[valid]
    if len(y_t) < 3 or np.all(y_t == y_t[0]) or np.all(y_p == y_p[0]):
        return np.nan, np.nan
    res = stats.spearmanr(y_t, y_p)
    return float(res.statistic), float(res.pvalue)


def calc_pearson(y_true, y_pred):
    """Compute Pearson correlation r and p-value."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_t = y_true[valid]
    y_p = y_pred[valid]
    if len(y_t) < 3 or np.all(y_t == y_t[0]) or np.all(y_p == y_p[0]):
        return np.nan, np.nan
    res = stats.pearsonr(y_t, y_p)
    return float(res.statistic), float(res.pvalue)
