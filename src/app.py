"""
app.py - FastAPI prediction service for Breast Cancer Classification.

Endpoints:
  GET  /          - API info
  GET  /health    - health check
  GET  /features  - list expected feature names
  POST /predict   - single prediction
  POST /predict/batch - batch prediction
"""

import os
import sys
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(__file__))
from predict import predict, predict_batch, _get_features

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Breast Cancer Classification API",
    description=(
        "MLOps-powered REST API for predicting breast cancer diagnosis "
        "(Benign / Malignant) using an MLflow-registered scikit-learn model."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Single prediction request — provide all 30 breast cancer feature values."""

    mean_radius: float = Field(..., example=14.12, description="Mean radius")
    mean_texture: float = Field(..., example=19.26, description="Mean texture")
    mean_perimeter: float = Field(..., example=91.98, description="Mean perimeter")
    mean_area: float = Field(..., example=654.9, description="Mean area")
    mean_smoothness: float = Field(..., example=0.0964, description="Mean smoothness")
    mean_compactness: float = Field(..., example=0.1006, description="Mean compactness")
    mean_concavity: float = Field(..., example=0.0889, description="Mean concavity")
    mean_concave_points: float = Field(..., example=0.0489, description="Mean concave points")
    mean_symmetry: float = Field(..., example=0.1812, description="Mean symmetry")
    mean_fractal_dimension: float = Field(..., example=0.0628, description="Mean fractal dimension")
    radius_error: float = Field(..., example=0.3001, description="Radius error")
    texture_error: float = Field(..., example=1.1890, description="Texture error")
    perimeter_error: float = Field(..., example=2.0930, description="Perimeter error")
    area_error: float = Field(..., example=24.63, description="Area error")
    smoothness_error: float = Field(..., example=0.0060, description="Smoothness error")
    compactness_error: float = Field(..., example=0.0197, description="Compactness error")
    concavity_error: float = Field(..., example=0.0254, description="Concavity error")
    concave_points_error: float = Field(..., example=0.0095, description="Concave points error")
    symmetry_error: float = Field(..., example=0.0209, description="Symmetry error")
    fractal_dimension_error: float = Field(..., example=0.0040, description="Fractal dimension error")
    worst_radius: float = Field(..., example=16.26, description="Worst radius")
    worst_texture: float = Field(..., example=25.68, description="Worst texture")
    worst_perimeter: float = Field(..., example=107.26, description="Worst perimeter")
    worst_area: float = Field(..., example=827.9, description="Worst area")
    worst_smoothness: float = Field(..., example=0.1323, description="Worst smoothness")
    worst_compactness: float = Field(..., example=0.2398, description="Worst compactness")
    worst_concavity: float = Field(..., example=0.2714, description="Worst concavity")
    worst_concave_points: float = Field(..., example=0.1288, description="Worst concave points")
    worst_symmetry: float = Field(..., example=0.2977, description="Worst symmetry")
    worst_fractal_dimension: float = Field(..., example=0.0742, description="Worst fractal dimension")


class PredictResponse(BaseModel):
    prediction: int
    label: str
    probability_benign: float
    probability_malignant: float
    model: str = "BreastCancerClassifier"


# ── Helper ────────────────────────────────────────────────────────────────────

# Map our snake_case field names to sklearn's feature names (with spaces)
_FIELD_MAP = {
    "mean_radius": "mean radius",
    "mean_texture": "mean texture",
    "mean_perimeter": "mean perimeter",
    "mean_area": "mean area",
    "mean_smoothness": "mean smoothness",
    "mean_compactness": "mean compactness",
    "mean_concavity": "mean concavity",
    "mean_concave_points": "mean concave points",
    "mean_symmetry": "mean symmetry",
    "mean_fractal_dimension": "mean fractal dimension",
    "radius_error": "radius error",
    "texture_error": "texture error",
    "perimeter_error": "perimeter error",
    "area_error": "area error",
    "smoothness_error": "smoothness error",
    "compactness_error": "compactness error",
    "concavity_error": "concavity error",
    "concave_points_error": "concave points error",
    "symmetry_error": "symmetry error",
    "fractal_dimension_error": "fractal dimension error",
    "worst_radius": "worst radius",
    "worst_texture": "worst texture",
    "worst_perimeter": "worst perimeter",
    "worst_area": "worst area",
    "worst_smoothness": "worst smoothness",
    "worst_compactness": "worst compactness",
    "worst_concavity": "worst concavity",
    "worst_concave_points": "worst concave points",
    "worst_symmetry": "worst symmetry",
    "worst_fractal_dimension": "worst fractal dimension",
}


def _map_request(req: PredictRequest) -> dict:
    """Convert snake_case Pydantic model to sklearn feature name dict."""
    raw = req.model_dump()
    return {_FIELD_MAP[k]: v for k, v in raw.items()}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    return {
        "name": "Breast Cancer Classification API",
        "version": "1.0.0",
        "description": "MLOps prediction service — Benign / Malignant classification",
        "endpoints": {
            "POST /predict": "Single prediction",
            "POST /predict/batch": "Batch predictions",
            "GET /features": "List expected feature names",
            "GET /health": "Health check",
            "GET /docs": "Interactive API docs (Swagger UI)",
        },
    }


@app.get("/health", tags=["Info"])
def health():
    return {"status": "ok", "model": "BreastCancerClassifier"}


@app.get("/features", tags=["Info"])
def features():
    """Return the list of all 30 expected feature names."""
    return {
        "feature_count": 30,
        "features": _get_features(),
        "note": "All features must be provided for a prediction.",
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict_single(request: PredictRequest):
    """
    Predict breast cancer diagnosis for a single sample.

    - **prediction**: 0 = Malignant, 1 = Benign
    - **label**: Human-readable diagnosis
    - **probability_benign / probability_malignant**: Confidence scores
    """
    try:
        feature_dict = _map_request(request)
        result = predict(feature_dict)
        return PredictResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch_endpoint(requests: list[PredictRequest]):
    """
    Predict breast cancer diagnosis for multiple samples at once.
    """
    try:
        feature_dicts = [_map_request(r) for r in requests]
        results = predict_batch(feature_dicts)
        return {"count": len(results), "predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
