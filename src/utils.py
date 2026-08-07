"""
utils.py - Data loading and preprocessing utilities for Breast Cancer Classification
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "breast_cancer.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_data(data_path: str = DATA_PATH) -> pd.DataFrame:
    """Load dataset from CSV file. If not found, generate from sklearn and save it."""
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"[utils] Loaded data from {data_path} — shape: {df.shape}")
    else:
        print(f"[utils] CSV not found at {data_path}. Generating from sklearn...")
        df = generate_and_save_data(data_path)
    return df


def generate_and_save_data(data_path: str = DATA_PATH) -> pd.DataFrame:
    """Generate Breast Cancer dataset from sklearn and persist to CSV."""
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    raw = load_breast_cancer()
    df = pd.DataFrame(raw.data, columns=raw.feature_names)
    df["target"] = raw.target  # 0 = malignant, 1 = benign
    df.to_csv(data_path, index=False)
    print(f"[utils] Saved dataset to {data_path} — shape: {df.shape}")
    return df


def preprocess(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
):
    """
    Split and optionally scale the dataset.

    Returns:
        X_train, X_test, y_train, y_test, scaler (or None if scale=False)
    """
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns
        )

    print(
        f"[utils] Train: {X_train.shape}, Test: {X_test.shape} | "
        f"Positive rate: train={y_train.mean():.2f}, test={y_test.mean():.2f}"
    )
    return X_train, X_test, y_train, y_test, scaler


def save_scaler(scaler, path: str = None):
    """Persist StandardScaler to disk."""
    if path is None:
        path = os.path.join(MODELS_DIR, "scaler.pkl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)
    print(f"[utils] Scaler saved to {path}")
    return path


def load_scaler(path: str = None):
    """Load StandardScaler from disk."""
    if path is None:
        path = os.path.join(MODELS_DIR, "scaler.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scaler not found at {path}")
    return joblib.load(path)


def get_feature_names() -> list:
    """Return the 30 feature names for the Breast Cancer dataset."""
    return list(load_breast_cancer().feature_names)


if __name__ == "__main__":
    print("Generating Breast Cancer dataset...")
    df = generate_and_save_data()
    print(f"Done! Shape: {df.shape}")
    print(f"Target distribution:\n{df['target'].value_counts()}")
