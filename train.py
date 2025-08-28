

import os, json, joblib, numpy as np, pandas as pd, mlflow, mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from datetime import datetime


MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
META_PATH  = os.getenv("META_PATH", "models/metadata.json")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "dummy_fraud_min")
SEED       = int(os.getenv("SEED", "42"))
MAX_ITER   = int(os.getenv("MAX_ITER", "200"))

# -----------------------------------------------------------------------------
# Data generator
# -----------------------------------------------------------------------------
def make_dummy(n: int = 2000, seed: int = SEED) -> pd.DataFrame:
    """
    Create a synthetic dataset for fraud detection.
    Fraud = amount > 1000 AND hour < 6.
    """
    np.random.seed(seed)
    df = pd.DataFrame({
        "amount": np.random.exponential(200, n),
        "oldbalanceOrg": np.random.uniform(0, 5000, n),
        "newbalanceOrig": np.random.uniform(0, 5000, n),
        "oldbalanceDest": np.random.uniform(0, 5000, n),
        "newbalanceDest": np.random.uniform(0, 5000, n),
        "hour": np.random.randint(0, 24, n),
    })
    df["fraud"] = ((df["amount"] > 1000) & (df["hour"] < 6)).astype(int)
    return df


def train():
    print("[train] generating dummy data…")
    df = make_dummy()
    print(f"[train] dataset size={len(df)}; fraud rate={df['fraud'].mean():.3f}")

    X, y = df.drop("fraud", axis=1), df["fraud"]
    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=MAX_ITER, random_state=SEED))
    ])


    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run():
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("max_iter", MAX_ITER)
        mlflow.log_param("seed", SEED)
        mlflow.log_param("n_samples", len(df))
        mlflow.log_param("fraud_rate", float(df["fraud"].mean()))

        print("[train] fitting pipeline…")
        pipe.fit(Xtr, ytr)

        # Evaluate
        proba = pipe.predict_proba(Xva)[:, 1]
        roc = roc_auc_score(yva, proba)
        pr  = average_precision_score(yva, proba)
        print(f"[train] ROC_AUC={roc:.3f}; PR_AUC={pr:.3f}")

        mlflow.log_metric("roc_auc", float(roc))
        mlflow.log_metric("pr_auc", float(pr))

        # Save model
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(pipe, MODEL_PATH)
        mlflow.sklearn.log_model(pipe, "model")
        print(f"[train] model saved → {MODEL_PATH}")

        # Save metadata
        meta = {
            "trained_at_utc": datetime.utcnow().isoformat() + "Z",
            "features": list(X.columns),
            "metrics": {"roc_auc": float(roc), "pr_auc": float(pr)},
            "threshold_default": float(os.getenv("THRESHOLD", "0.5")),
        }
        os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        mlflow.log_artifact(META_PATH)


if __name__ == "__main__":
    train()
