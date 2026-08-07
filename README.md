# Breast Cancer MLOps Project

An end-to-end MLOps pipeline for **Breast Cancer Classification** (Benign vs. Malignant) using scikit-learn, MLflow, DVC, FastAPI, Docker, and GitHub Actions.

---

## 🏗️ Project Structure

```
.
├── data/                        # Dataset (versioned with DVC)
│   └── breast_cancer.csv
├── models/                      # Trained model artifacts
├── src/
│   ├── train.py                 # Trains 3 models + MLflow tracking
│   ├── app.py                   # FastAPI prediction service
│   ├── predict.py               # Inference helper (loads MLflow model)
│   └── utils.py                 # Data loading & preprocessing utilities
├── tests/
│   ├── test_api.py              # FastAPI endpoint tests
│   └── test_model.py            # Model + data pipeline unit tests
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI/CD pipeline
├── Dockerfile                   # Multi-stage Docker image
├── requirements.txt
├── dvc.yaml                     # DVC pipeline definition
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Initialize Git & DVC
```bash
git init
dvc init
```

### 3. Generate dataset & track with DVC
```bash
python src/utils.py          # generates data/breast_cancer.csv
dvc add data/breast_cancer.csv
git add data/breast_cancer.csv.dvc .gitignore
git commit -m "Add dataset with DVC tracking"
```

### 4. Train models
```bash
python src/train.py
```
This trains Logistic Regression, Random Forest, and Gradient Boosting — logs all experiments to MLflow and registers the best model.

### 5. View MLflow experiments
```bash
mlflow ui --backend-store-uri ./mlruns
```
Open http://localhost:5000 in your browser.

### 6. Run the FastAPI server
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```
Open http://localhost:8000/docs for the interactive Swagger UI.

### 7. Make a prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "mean_radius": 14.12,
    "mean_texture": 19.26,
    "mean_perimeter": 91.98,
    "mean_area": 654.9,
    "mean_smoothness": 0.0964,
    "mean_compactness": 0.1006,
    "mean_concavity": 0.0889,
    "mean_concave_points": 0.0489,
    "mean_symmetry": 0.1812,
    "mean_fractal_dimension": 0.0628,
    "radius_error": 0.3001,
    "texture_error": 1.189,
    "perimeter_error": 2.093,
    "area_error": 24.63,
    "smoothness_error": 0.006,
    "compactness_error": 0.0197,
    "concavity_error": 0.0254,
    "concave_points_error": 0.0095,
    "symmetry_error": 0.0209,
    "fractal_dimension_error": 0.004,
    "worst_radius": 16.26,
    "worst_texture": 25.68,
    "worst_perimeter": 107.26,
    "worst_area": 827.9,
    "worst_smoothness": 0.1323,
    "worst_compactness": 0.2398,
    "worst_concavity": 0.2714,
    "worst_concave_points": 0.1288,
    "worst_symmetry": 0.2977,
    "worst_fractal_dimension": 0.0742
  }'
```

### 8. Run tests
```bash
pytest tests/ -v
```

### 9. Build & run Docker
```bash
docker build -t breast-cancer-api .
docker run -p 8000:8000 breast-cancer-api
```

---

## 📊 Models Trained

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Baseline linear classifier |
| **Random Forest** | Ensemble of 200 decision trees |
| **Gradient Boosting** | Boosted ensemble (150 estimators) |

The best-performing model (by F1-score) is automatically registered in the MLflow Model Registry as **`BreastCancerClassifier`** and promoted to **Production** stage.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API info |
| `GET` | `/health` | Health check |
| `GET` | `/features` | List all 30 feature names |
| `POST` | `/predict` | Single prediction |
| `POST` | `/predict/batch` | Batch predictions |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc UI |

---

## 🔄 CI/CD Pipeline (GitHub Actions)

The workflow in `.github/workflows/ci.yml` runs on every push/PR:

1. **Test job** — sets up Python, installs deps, trains models, runs pytest
2. **Docker job** — builds the Docker image and smoke-tests the container

---

## 📋 Dataset

**Breast Cancer Wisconsin Dataset** from scikit-learn:
- 569 samples, 30 numerical features
- Binary classification: 0 = Malignant, 1 = Benign
- Source: UCI Machine Learning Repository

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| ML | scikit-learn |
| Experiment Tracking | MLflow |
| API | FastAPI + Uvicorn |
| Data Versioning | DVC |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Testing | pytest |
