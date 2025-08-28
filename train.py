import os, joblib, numpy as np, pandas as pd, mlflow, mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

MODEL_PATH = "models/model.pkl"
EXPERIMENT = "dummy_fraud_min"

def make_dummy(n=2000, seed=42):
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
    df = make_dummy()
    X, y = df.drop("fraud", axis=1), df["fraud"]
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipe = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=200))])

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run():
        mlflow.log_param("model", "LogReg")
        mlflow.log_param("max_iter", 200)
        pipe.fit(Xtr, ytr)
        proba = pipe.predict_proba(Xva)[:, 1]
        roc = roc_auc_score(yva, proba)
        pr = average_precision_score(yva, proba)
        mlflow.log_metric("roc_auc", float(roc))
        mlflow.log_metric("pr_auc", float(pr))
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(pipe, MODEL_PATH)
        mlflow.sklearn.log_model(pipe, "model")
        print(f"Saved {MODEL_PATH}; ROC_AUC={roc:.3f}; PR_AUC={pr:.3f}")

if __name__ == "__main__":
    train()