import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict():
    sample = {
        "features": {
            "checking_status": "<0",
            "duration": 12,
            "credit_history": "existing paid",
            "purpose": "radio/tv",
            "credit_amount": 2000,
            "savings_status": "<100",
            "employment": "1<=X<4",
            "installment_commitment": 2,
            "personal_status": "male single",
            "other_parties": "none",
            "residence_since": 2,
            "property_magnitude": "car",
            "age": 35,
            "other_payment_plans": "none",
            "housing": "own",
            "existing_credits": 1,
            "job": "skilled",
            "num_dependents": 1,
            "own_telephone": "yes",
            "foreign_worker": "yes"
        }
    }
    r = client.post("/predict", json=sample)
    assert r.status_code == 200
    data = r.json()
    assert 0.0 <= data["probability_default"] <= 1.0
    assert data["risk_level"] in ["low", "medium", "high"]