"""
predict.py - Model inference helper.

Loads the registered Production model from the MLflow Model Registry
and provides a predict() function used by the FastAPI app.
"""

import os
import sys
import json
import numpy as np
import mlflow
import mlflow.sklearn
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_scaler, get_feature_names

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{(_PROJECT_ROOT / 'mlflow.db').as_posix()}",
)
REGISTERED_MODEL_NAME = "BreastCancerClassifier"
MODELS_DIR = str(_PROJECT_ROOT / "models")

# ── Lazy-loaded globals ───────────────────────────────────────────────────────
_model = None
_scaler = None
_feature_names = None


def _load_model():
    """Load the Production model from the MLflow Model Registry (with fallback)."""
    global _model
    if _model is not None:
        return _model

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/Production"
    try:
        _model = mlflow.sklearn.load_model(model_uri)
        print(f"[predict] Loaded Production model from: {model_uri}")
    except Exception as e:
        # Fallback: try loading the latest version regardless of stage
        print(f"[predict] Production stage load failed ({e}). Trying latest version...")
        try:
            model_uri = f"models:/{REGISTERED_MODEL_NAME}/latest"
            _model = mlflow.sklearn.load_model(model_uri)
            print(f"[predict] Loaded latest model from registry.")
        except Exception as e2:
            # Last fallback: load from training summary
            print(f"[predict] Registry load failed ({e2}). Loading from summary...")
            summary_path = os.path.join(MODELS_DIR, "training_summary.json")
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    summary = json.load(f)
                run_id = summary["best_run_id"]
                _model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
                print(f"[predict] Loaded model from run {run_id}")
            else:
                raise RuntimeError(
                    "No trained model found. Please run src/train.py first."
                )
    return _model


def _load_scaler():
    global _scaler
    if _scaler is None:
        _scaler = load_scaler()
    return _scaler


def _get_features():
    global _feature_names
    if _feature_names is None:
        _feature_names = get_feature_names()
    return _feature_names


def predict(features: dict) -> dict:
    """
    Run inference on a single sample.

    Args:
        features: dict mapping feature names → float values
                  (all 30 breast cancer features required)

    Returns:
        dict with:
          - prediction: int (0 = Malignant, 1 = Benign)
          - label: str
          - probability_benign: float
          - probability_malignant: float
    """
    model = _load_model()
    scaler = _load_scaler()
    feature_names = _get_features()

    # Build ordered feature vector
    X = np.array([[features[fn] for fn in feature_names]])

    # Scale
    X_scaled = scaler.transform(X)

    # Predict
    prediction = int(model.predict(X_scaled)[0])
    proba = model.predict_proba(X_scaled)[0]

    return {
        "prediction": prediction,
        "label": "Benign" if prediction == 1 else "Malignant",
        "probability_benign": round(float(proba[1]), 4),
        "probability_malignant": round(float(proba[0]), 4),
    }


def predict_batch(records: list) -> list:
    """Run inference on a list of feature dicts."""
    return [predict(r) for r in records]


if __name__ == "__main__":
    # Quick smoke test with mean values from the Breast Cancer dataset
    from sklearn.datasets import load_breast_cancer
    raw = load_breast_cancer()
    mean_features = dict(zip(raw.feature_names, raw.data.mean(axis=0).tolist()))
    result = predict(mean_features)
    print("Smoke test result:", result)
