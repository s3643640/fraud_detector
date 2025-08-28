from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os, datetime as dt, pandas as pd, joblib


from app.model import loaded, predict_proba_row

THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))

class Txn(BaseModel):
    amount: float = Field(..., ge=0)
    oldbalanceOrg: float = Field(..., ge=0)
    newbalanceOrig: float = Field(..., ge=0)
    oldbalanceDest: float = Field(..., ge=0)
    newbalanceDest: float = Field(..., ge=0)
    hour: int = Field(..., ge=0, le=23)

app = FastAPI(title="Fraud Detection API", version="1.0")

@app.get("/health")
def health():
    return {"status": "ok" if loaded() else "model_missing"}

@app.post("/predict")
def predict(txn: Txn):
    if not loaded():
        raise HTTPException(503, "Model not loaded; run train.py first")
    proba = predict_proba_row(txn.dict())
    return {
        "fraud_probability": proba,
        "is_fraud": proba >= THRESHOLD,
        "threshold": THRESHOLD,
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
    }
