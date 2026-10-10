from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "heart_disease_model.joblib"

app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0.0",
    description="Academic demonstration of a binary heart disease classifier."
)

model = joblib.load(MODEL_PATH)

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

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        features = request.model_dump()
        row = pd.DataFrame([features])
        prediction = int(model.predict(row)[0])
        probabilities = model.predict_proba(row)[0]

        return {
            "prediction": prediction,
            "label": (
                "heart_disease"
                if prediction == 1
                else "no_heart_disease"
            ),
            "confidence": float(max(probabilities)),
            "disease_probability": float(probabilities[1]),
            "model_file": MODEL_PATH.name
        }
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {exc}"
        ) from exc
