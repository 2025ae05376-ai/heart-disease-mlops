# Heart Disease Classification — MLOps Assignment 1

An educational MLOps project using the UCI Heart Disease Cleveland dataset, scikit-learn, MLflow, FastAPI, Docker, GitHub Actions, Minikube/Kubernetes and Prometheus-compatible metrics.

## Project structure

- `data/heart_disease_raw_with_target.csv` — prepared dataset
- `src/download_data.py` — download and prepare UCI dataset ID 45
- `src/train.py` — preprocessing, cross-validation, model selection and evaluation
- `notebooks/01_EDA_Model_Development.ipynb` — EDA and model-development workflow
- `models/` — serialized model pipeline and evaluation metrics
- `mlflow_artifacts/` — evaluation plots and local MLflow artifacts
- `api/app.py` — prediction API, request logging and metrics endpoint
- `tests/` — API and training/data tests
- `.github/workflows/ci.yml` — automated lint, test and training workflow
- `Dockerfile` — container definition
- `k8s/deployment.yaml` — Kubernetes Deployment and Service
- `monitoring/prometheus.yml` — Prometheus scrape configuration

## Setup

Use Python 3.12, consistent with the Docker image and CI workflow.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data acquisition

With network access, regenerate the dataset using:

```bash
python src/download_data.py
```

This fetches UCI dataset ID 45, preserves its original target as `num` and `target_original`, and creates binary `target`: 0 means no disease and 1 means disease present. The current dataset has 303 rows; missing values in `ca` and `thal` are handled by the training pipeline.

## Training and evaluation

```bash
python src/train.py
```

The script compares Logistic Regression and Random Forest using stratified five-fold cross-validation ROC-AUC on the training split. It evaluates the selected pipeline on a held-out test split and saves `models/heart_disease_model.joblib` and `models/metrics.json`.

## Tests

```bash
python -m pytest -q
ruff check api src tests --select E9,F63,F7,F82
```

GitHub Actions runs linting, tests, model training and artifact checks on pushes and pull requests to `main`.

## MLflow

The `heart-disease-classification-linux` experiment was created with a Linux-compatible artifact location. The verified Logistic Regression and Random Forest runs each contain a confusion matrix, ROC curve and `metrics.json` under `evaluation/`.

To launch the local UI from the repository root:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
```

Use the lab environment's port-access feature to open the UI. Do not assume it is publicly accessible.

## API

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

- `GET /health` — health status
- `POST /predict` — predicted class, label, confidence and disease probability
- `GET /metrics` — Prometheus-format request counters and latency histograms

Example:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"age":55,"sex":1,"cp":0,"trestbps":130,"chol":240,"fbs":0,"restecg":1,"thalach":150,"exang":0,"oldpeak":1.0,"slope":1,"ca":0,"thal":2}'
```

Request logs record method, path, status and duration, not submitted feature values.

## Docker

```bash
docker build -t heart-disease-api:v2 .
docker run --rm -p 8001:8000 heart-disease-api:v2
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8001/metrics
```

## Local Kubernetes deployment

The API was previously verified on Minikube in the lab environment. This is a local cluster deployment, not a public cloud deployment.

```bash
minikube status
minikube kubectl -- get deployment,pods,services
minikube kubectl -- rollout status deployment/heart-disease-api
```

The NodePort and cluster IP can vary. Obtain current values using `minikube ip` and `minikube kubectl -- get service heart-disease-api`.

## Monitoring and submission evidence

The API exposes Prometheus-compatible counters and latency histograms. `monitoring/prometheus.yml` is a scrape configuration template.

## Report PDF of MLOps Assignment 1

https://drive.google.com/file/d/1rUtM41fVHKnPF7aFNX2iD6p8a2mivEBa/view?usp=sharing

## Video Link of Demonstration

https://drive.google.com/file/d/10i6eIjubRMZF1zun60Hqz4na3aeHUGW1/view 
