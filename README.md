# 🚀 Fraud Detection API (FastAPI + Scikit-learn + MLflow + Docker)

This repository contains a **machine learning application** that detects potentially fraudulent transactions with endpoints accessed via an inference server. 
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
- Once the Fraud Detection API page loads, click the "Try it out" button and copy the json request below into the Request Body.
- Click Execute
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
- You can see the probability that the transaction is a fraudulent one and the detection as to true or not in the Response body.
## 🖥️ Running with Docker
### 1. Build the image
  ```bash
  docker build -t fraud-api .
  ```

### 2. Run the container
  ```bash
  docker run -p 8000:8000 fraud-api
  ```
### 3. Test the API
  - Swagger UI available at: http://127.0.0.1:8000/docs

## 📊 ML Flow
- ### This project logs parameters, metrics, and artifacts to MLflow.
  Start MLflow UI
  ```bash
  mlflow ui
  ```
  Open in browser

  👉 http://127.0.0.1:5000

  You’ll see:

  Logged parameters (e.g., max_iter, n_samples, fraud_rate)

  Validation metrics (ROC AUC, PR AUC)

  Saved artifacts (model.pkl)
