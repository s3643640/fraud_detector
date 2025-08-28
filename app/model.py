
import os, joblib, pandas as pd

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
_model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

def loaded():
    return _model is not None

def predict_proba_row(features: dict):
    if _model is None:
        raise RuntimeError("Model not loaded")
    X = pd.DataFrame([features])
    return float(_model.predict_proba(X)[:, 1][0])