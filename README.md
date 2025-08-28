# 🚀 Fraud Detection API (FastAPI + Scikit-learn + MLflow + Docker)

This repository contains a **machine learning application** that detects potentially fraudulent transactions.  
It uses **FastAPI** to serve predictions, **scikit-learn** for modeling, **MLflow** for experiment tracking, and **Docker** for containerization.


---

## ⚙️ Features
- **Synthetic dataset** of transactions (amount, balances, time of day).
- **Fraud labeling rule**: transactions > $1000 before 6am are "fraud".
- **Logistic Regression pipeline** with feature scaling.
- **MLflow tracking** of hyperparameters, metrics (ROC AUC, PR AUC), and artifacts.
- **FastAPI endpoints**:
  - `GET /health` – check if model is loaded.
  - `POST /predict` – predict fraud probability for a transaction.

---

## 🖥️ Running Locally

### 1. Install dependencies, train and Run the API
```bash
python -m pip install -r requirements.txt
python train.py
python -m uvicorn app.main:app --reload
```
- Swagger UI: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health
### 2. Example Request
```json
{
  "amount": 1200,
  "oldbalanceOrg": 3000,
  "newbalanceOrig": 1800,
  "oldbalanceDest": 500,
  "newbalanceDest": 1700,
  "hour": 2
}
```
### 3. Example Output
```json
{
  "fraud_probability": 0.82,
  "is_fraud": true,
  "threshold": 0.5,
  "timestamp": "2025-08-28T12:00:00Z"
}
```
