"""
Integrity Tests: In-Fold Transformation Strictness.
Verifies that feature scalers and models are refit strictly on train subsets and not influenced by test fold distributions.
"""

import numpy as np
import pandas as pd
import pytest
from src.models.linear_pipeline import InFoldLinearModel


def test_infold_scaler_isolation():
    # Construct synthetic check dataframe with extreme outlier only in test fold
    np.random.seed(42)
    genes = [f"GENE_{i}" for i in range(50)]
    X_train = pd.DataFrame(np.random.normal(0, 1, size=(20, 50)), columns=genes)
    y_train = np.random.binomial(1, 0.5, size=20)
    
    X_test = pd.DataFrame(np.random.normal(100, 1, size=(5, 50)), columns=genes)

    model = InFoldLinearModel(model_type="ridge_classifier", top_hvg=20)
    model.fit_transform_train(X_train, y_train)

    # Scaler mean must be near 0 (train distribution), unaffected by test set mean of 100
    assert np.all(np.abs(model.scaler.mean_) < 1.0), "Scaler was contaminated by test distribution!"
    
    preds = model.predict(X_test)
    assert len(preds) == 5
    assert not np.any(np.isnan(preds))
