"""
train.py - MLflow-tracked training pipeline for Breast Cancer Classification.

Trains 3 models, logs params/metrics/artifacts, and registers the best model
in the MLflow Model Registry.

Supports two backends:
  - DagsHub (remote): set DAGSHUB_TOKEN env variable
  - SQLite (local/CI): automatic fallback
"""

import os
import sys
import json
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_data, generate_and_save_data, preprocess, save_scaler, DATA_PATH, MODELS_DIR

# ── DagsHub / MLflow Configuration ───────────────────────────────────────────
DAGSHUB_USERNAME = "akkshay0312"
DAGSHUB_REPO     = "Capstone-Project---MLOps-Pipeline"
DAGSHUB_TOKEN    = os.environ.get("DAGSHUB_TOKEN", "")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXPERIMENT_NAME      = "BreastCancerClassification"
REGISTERED_MODEL_NAME = "BreastCancerClassifier"


def _setup_mlflow():
    """
    Configure MLflow tracking:
      - If DAGSHUB_TOKEN is set  → use DagsHub remote tracking server
      - Otherwise               → fall back to local SQLite
    """
    if DAGSHUB_TOKEN:
        print("[mlflow] Using DagsHub remote tracking...")
        try:
            import dagshub
            dagshub.init(
                repo_owner=DAGSHUB_USERNAME,
                repo_name=DAGSHUB_REPO,
                mlflow=True,
            )
            print(f"[mlflow] Tracking URI: https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")
            return
        except Exception as e:
            print(f"[mlflow] DagsHub init failed ({e}), falling back to SQLite.")

    # Fallback: local SQLite backend
    sqlite_uri = f"sqlite:///{(_PROJECT_ROOT / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(sqlite_uri)
    print(f"[mlflow] Using local SQLite tracking: {sqlite_uri}")

EXPERIMENT_NAME      = "BreastCancerClassification"
REGISTERED_MODEL_NAME = "BreastCancerClassifier"

# ── Model Definitions ─────────────────────────────────────────────────────────
MODELS = {
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "params": {
            "max_iter": 1000,
            "solver": "lbfgs",
            "C": 1.0,
            "random_state": 42,
        },
    },
    "RandomForest": {
        "model": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        ),
        "params": {
            "n_estimators": 200,
            "max_depth": 10,
            "random_state": 42,
        },
    },
    "GradientBoosting": {
        "model": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42
        ),
        "params": {
            "n_estimators": 150,
            "learning_rate": 0.1,
            "max_depth": 4,
            "random_state": 42,
        },
    },
}


def evaluate(model, X_test, y_test) -> dict:
    """Compute classification metrics for a fitted model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }


def run_training():
    # ── Setup ─────────────────────────────────────────────────────────────────
    _setup_mlflow()
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load / generate data
    if not os.path.exists(DATA_PATH):
        generate_and_save_data()
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess(df)

    # Persist scaler
    os.makedirs(MODELS_DIR, exist_ok=True)
    scaler_path = save_scaler(scaler)

    best_run_id = None
    best_f1 = -1.0
    results = []

    # ── Train each model ──────────────────────────────────────────────────────
    for model_name, config in MODELS.items():
        print(f"\n{'='*60}")
        print(f"  Training: {model_name}")
        print(f"{'='*60}")

        with mlflow.start_run(run_name=model_name) as run:
            # Log params
            mlflow.log_params(config["params"])
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("dataset", "breast_cancer")
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("scaler", "StandardScaler")

            # Train
            model = config["model"]
            model.fit(X_train, y_train)

            # Evaluate
            metrics = evaluate(model, X_test, y_test)
            mlflow.log_metrics(metrics)

            # Log the classification report as a text artifact
            report = classification_report(
                y_test, model.predict(X_test), target_names=["Malignant", "Benign"]
            )
            report_path = os.path.join(MODELS_DIR, f"{model_name}_report.txt")
            with open(report_path, "w") as f:
                f.write(f"Model: {model_name}\n\n{report}")
            mlflow.log_artifact(report_path)

            # Log scaler artifact
            mlflow.log_artifact(scaler_path)

            # Log model artifact
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=None,  # register best only
            )

            run_id = run.info.run_id
            print(f"  Run ID : {run_id}")
            for k, v in metrics.items():
                print(f"  {k:12s}: {v:.4f}")

            results.append(
                {"model": model_name, "run_id": run_id, **metrics}
            )

            if metrics["f1_score"] > best_f1:
                best_f1 = metrics["f1_score"]
                best_run_id = run_id
                best_model_name = model_name

    # ── Register best model ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Best model: {best_model_name} (F1={best_f1:.4f})")
    print(f"  Registering as '{REGISTERED_MODEL_NAME}' ...")
    print(f"{'='*60}")

    model_uri = f"runs:/{best_run_id}/model"
    try:
        mv = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
        print(f"  Registered version: {mv.version}")

        # Transition to Production
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=mv.version,
            stage="Production",
        )
        print(f"  Model transitioned to Production.")
    except Exception as e:
        print(f"  Model registry step skipped (expected on DagsHub S3 remote): {e}")

    # Save summary
    summary = {
        "best_model": best_model_name,
        "best_run_id": best_run_id,
        "best_f1": best_f1,
        "registered_model": REGISTERED_MODEL_NAME,
        "registered_version": mv.version,
        "all_results": results,
    }
    summary_path = os.path.join(MODELS_DIR, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {summary_path}")

    # Print results table
    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<25} {'Accuracy':>10} {'F1':>10} {'ROC-AUC':>10}")
    print(f"  {'-'*55}")
    for r in results:
        marker = " [BEST]" if r["model"] == best_model_name else ""
        print(
            f"  {r['model']:<25} {r['accuracy']:>10.4f} {r['f1_score']:>10.4f} {r['roc_auc']:>10.4f}{marker}"
        )
    print()

    return summary


if __name__ == "__main__":
    run_training()
