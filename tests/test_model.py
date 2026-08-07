"""
test_model.py - Unit tests for the ML model and data pipeline.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_load_data():
    """Dataset should load with correct shape and target column."""
    from utils import load_data, generate_and_save_data, DATA_PATH
    if not os.path.exists(DATA_PATH):
        generate_and_save_data()
    df = load_data()
    assert df.shape[0] > 0, "DataFrame should have rows"
    assert "target" in df.columns, "target column must exist"
    assert df.shape[1] == 31, "Should have 30 features + 1 target"


def test_preprocess_split():
    """Preprocess should return correct train/test shapes."""
    from utils import load_data, generate_and_save_data, preprocess, DATA_PATH
    if not os.path.exists(DATA_PATH):
        generate_and_save_data()
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess(df, test_size=0.2)
    total = len(df)
    assert len(X_train) + len(X_test) == total
    assert len(y_train) + len(y_test) == total
    assert scaler is not None, "Scaler should be returned"


def test_preprocess_no_nan():
    """Scaled features should contain no NaN or Inf values."""
    from utils import load_data, generate_and_save_data, preprocess, DATA_PATH
    if not os.path.exists(DATA_PATH):
        generate_and_save_data()
    df = load_data()
    X_train, X_test, _, _, _ = preprocess(df)
    assert not np.isnan(X_train.values).any(), "No NaN in training features"
    assert not np.isinf(X_train.values).any(), "No Inf in training features"


def test_feature_names():
    """Should return exactly 30 feature names."""
    from utils import get_feature_names
    names = get_feature_names()
    assert len(names) == 30
    assert "mean radius" in names


def test_predict_output_shape():
    """predict() should return a dict with all required keys."""
    from predict import predict
    from utils import get_feature_names
    from sklearn.datasets import load_breast_cancer

    raw = load_breast_cancer()
    mean_vals = dict(zip(raw.feature_names, raw.data.mean(axis=0).tolist()))

    result = predict(mean_vals)
    assert "prediction" in result
    assert "label" in result
    assert "probability_benign" in result
    assert "probability_malignant" in result
    assert result["prediction"] in (0, 1)
    assert result["label"] in ("Benign", "Malignant")


def test_predict_probabilities_valid():
    """Probabilities should be between 0 and 1 and sum to 1."""
    from predict import predict
    from sklearn.datasets import load_breast_cancer

    raw = load_breast_cancer()
    mean_vals = dict(zip(raw.feature_names, raw.data.mean(axis=0).tolist()))

    result = predict(mean_vals)
    p_b = result["probability_benign"]
    p_m = result["probability_malignant"]
    assert 0.0 <= p_b <= 1.0
    assert 0.0 <= p_m <= 1.0
    assert abs(p_b + p_m - 1.0) < 0.01
