import logging
logging.basicConfig(level=logging.INFO)
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict

MODEL_PATH = "app/model/model.joblib"

app = FastAPI(title="Credit Risk ML API", version="1.0")

model = joblib.load(MODEL_PATH)

class PredictRequest(BaseModel):
    # enviamos features como dict flexible (más fácil para demo)
    features: Dict[str, Any]

class PredictResponse(BaseModel):
    probability_default: float
    risk_level: str

@app.get("/health")
def health():
    return {"status": "ok"}

def risk_bucket(p: float) -> str:
    # simple thresholds para demo
    if p < 0.33:
        return "low"
    elif p < 0.66:
        return "medium"
    return "high"

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    df = pd.DataFrame([req.features])
    proba = float(model.predict_proba(df)[:, 1][0])

    logging.info(
    f"Prediction made | Probability: {proba:.4f} | Risk: {risk_bucket(proba)}"
)

    logging.info(f"Prediction requested. Probability: {proba}")

    return {
        "probability_default": round(proba, 4),
        "risk_level": risk_bucket(proba)
    }

@app.get("/metadata")
def metadata():
    return {
        "model_type": "Logistic Regression",
        "target": "credit default (1 = bad, 0 = good)",
        "version": "1.0",
        "description": "Credit risk scoring model trained on German Credit dataset"
    }