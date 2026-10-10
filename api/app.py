from pathlib import Path
import logging
import time

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, ConfigDict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("heart_disease_api")

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "heart_disease_model.joblib"

app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.1.0",
    description="Academic demonstration of a binary heart disease classifier."
)

model = joblib.load(MODEL_PATH)

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"]
)

class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: float | None = None
    thal: float | None = None

@app.middleware("http")
async def record_metrics(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        duration = time.perf_counter() - start
        logger.exception(
            "request_failed method=%s path=%s duration_seconds=%.4f",
            request.method, request.url.path, duration
        )
        raise
    duration = time.perf_counter() - start
    logger.info(
        "request_completed method=%s path=%s status=%s duration_seconds=%.4f",
        request.method, request.url.path, response.status_code, duration
    )

    REQUEST_COUNT.labels(
        request.method, request.url.path, str(response.status_code)
    ).inc()
    REQUEST_LATENCY.labels(
        request.method, request.url.path
    ).observe(duration)

    return response

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        row = pd.DataFrame([request.model_dump()])
        prediction = int(model.predict(row)[0])
        probabilities = model.predict_proba(row)[0]

        return {
            "prediction": prediction,
            "label": "heart_disease" if prediction == 1 else "no_heart_disease",
            "confidence": float(max(probabilities)),
            "disease_probability": float(probabilities[1]),
            "model_file": MODEL_PATH.name
        }
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {exc}"
        ) from exc
