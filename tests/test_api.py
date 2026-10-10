from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

SAMPLE = {
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1
}

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True

def test_prediction_endpoint():
    response = client.post("/predict", json=SAMPLE)
    assert response.status_code == 200
    result = response.json()
    assert result["prediction"] in [0, 1]
    assert 0 <= result["disease_probability"] <= 1
    assert 0 <= result["confidence"] <= 1

def test_invalid_request():
    response = client.post("/predict", json={"age": 63})
    assert response.status_code == 422
