"""
test_api.py - FastAPI endpoint tests using TestClient.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture(scope="module")
def client():
    from app import app
    return TestClient(app)


# ── Sample benign data (from dataset index 0) ─────────────────────────────────
SAMPLE_BENIGN = {
    "mean_radius": 17.99,
    "mean_texture": 10.38,
    "mean_perimeter": 122.80,
    "mean_area": 1001.0,
    "mean_smoothness": 0.1184,
    "mean_compactness": 0.2776,
    "mean_concavity": 0.3001,
    "mean_concave_points": 0.1471,
    "mean_symmetry": 0.2419,
    "mean_fractal_dimension": 0.07871,
    "radius_error": 1.095,
    "texture_error": 0.9053,
    "perimeter_error": 8.589,
    "area_error": 153.4,
    "smoothness_error": 0.006399,
    "compactness_error": 0.04904,
    "concavity_error": 0.05373,
    "concave_points_error": 0.01587,
    "symmetry_error": 0.03003,
    "fractal_dimension_error": 0.006193,
    "worst_radius": 25.38,
    "worst_texture": 17.33,
    "worst_perimeter": 184.6,
    "worst_area": 2019.0,
    "worst_smoothness": 0.1622,
    "worst_compactness": 0.6656,
    "worst_concavity": 0.7119,
    "worst_concave_points": 0.2654,
    "worst_symmetry": 0.4601,
    "worst_fractal_dimension": 0.1189,
}


def test_root(client):
    """GET / should return 200 with API info."""
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "name" in data
    assert "endpoints" in data


def test_health(client):
    """GET /health should return status ok."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_features(client):
    """GET /features should return 30 features."""
    resp = client.get("/features")
    assert resp.status_code == 200
    data = resp.json()
    assert data["feature_count"] == 30
    assert len(data["features"]) == 30


def test_predict_returns_valid_response(client):
    """POST /predict should return a valid prediction dict."""
    resp = client.post("/predict", json=SAMPLE_BENIGN)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "prediction" in data
    assert "label" in data
    assert data["prediction"] in (0, 1)
    assert data["label"] in ("Benign", "Malignant")
    assert 0.0 <= data["probability_benign"] <= 1.0
    assert 0.0 <= data["probability_malignant"] <= 1.0


def test_predict_probabilities_sum_to_one(client):
    """Benign + Malignant probabilities should sum to ~1."""
    resp = client.post("/predict", json=SAMPLE_BENIGN)
    assert resp.status_code == 200
    data = resp.json()
    total = data["probability_benign"] + data["probability_malignant"]
    assert abs(total - 1.0) < 0.01


def test_predict_missing_field(client):
    """POST /predict with a missing field should return 422."""
    incomplete = {k: v for k, v in SAMPLE_BENIGN.items() if k != "mean_radius"}
    resp = client.post("/predict", json=incomplete)
    assert resp.status_code == 422


def test_predict_batch(client):
    """POST /predict/batch should return results for each sample."""
    resp = client.post("/predict/batch", json=[SAMPLE_BENIGN, SAMPLE_BENIGN])
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2
