# BankSim Fraud Detection FastAPI

Production-ready FastAPI application for real-time fraud scoring using the trained **Isolation Forest** model from the BankSim ML pipeline.

---

## Overview

This API:
- Accepts a single raw transaction (JSON)
- Computes the **17 features** used during model training
- Loads the trained **Isolation Forest** model
- Returns fraud prediction, anomaly score, risk level, and human-readable reasons
- Is designed for real-time inference with <100ms latency

---

## Project Structure

```
anamoly/
├── app/
│   ├── __init__.py
│   ├── main.py                      # FastAPI application entry point
│   ├── schemas.py                   # Pydantic request/response models
│   └── utils_feature_engineering.py # Feature computation logic
├── data/
│   ├── raw/
│   │   └── banksim.csv              # Original BankSim data
│   └── processed/
│       ├── banksim_feature_engineered.csv
│       ├── X_features.csv
│       └── y_labels.csv
├── models/
│   ├── isolation_forest_model.pkl   # Trained Isolation Forest model
│   ├── one_class_svm_model.pkl      # Trained OCSVM model
│   └── ocsvm_scaler.pkl             # Scaler for OCSVM
├── tests/
│   ├── test_api_setup.py            # Artifact + model smoke tests
│   └── test_api_routes.py           # HTTP endpoint tests
└── docs/
    └── README_API.md                # This file
```

---

## Installation

### Prerequisites
- Python 3.8+
- Trained Isolation Forest model (`isolation_forest_model.pkl`)
- Engineered training data (`banksim_feature_engineered.csv`)

### Install Dependencies

```bash
pip install fastapi uvicorn pydantic numpy pandas scikit-learn joblib
```

Or create a `requirements.txt`:

```txt
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.3
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
joblib==1.3.2
```

Then install:

```bash
pip install -r requirements.txt
```

---

## Running the API

### Start the server

```bash
uvicorn app.main:app --reload
```

Expected output:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345] using StatReload
INFO:     Started server process [12346]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### Access the API

- **Interactive Docs (Swagger):** http://127.0.0.1:8000/docs
- **Alternative Docs (ReDoc):** http://127.0.0.1:8000/redoc
- **Health Check:** http://127.0.0.1:8000/health

---

## API Endpoints

### 1. Health Check

**GET /health**

Check if the API is running.

**Response:**
```json
{
  "status": "ok"
}
```

**cURL:**
```bash
curl http://127.0.0.1:8000/health
```

---

### 2. Fraud Prediction

**POST /predict**

Score a single transaction for fraud risk.

**Request Body (JSON):**
```json
{
  "customer": "C1093826151",
  "age": "4",
  "gender": "M",
  "zipcodeOri": "28007",
  "merchant": "M348934600",
  "zipMerchant": "28007",
  "category": "es_transportation",
  "amount": 50.0,
  "step": 100
}
```

**Field Descriptions:**
| Field | Type | Description |
|-------|------|-------------|
| `customer` | string | Unique customer identifier |
| `age` | string | Age bucket (e.g., '1', '2', '3', '4') |
| `gender` | string | Gender ('M' or 'F') |
| `zipcodeOri` | string | Customer's zipcode |
| `merchant` | string | Merchant identifier |
| `zipMerchant` | string | Merchant's zipcode |
| `category` | string | Merchant category (e.g., 'es_transportation') |
| `amount` | float | Transaction amount in euros (≥0) |
| `step` | int | Time step (hours since simulation start, ≥0) |

**Response:**
```json
{
  "is_fraud": false,
  "fraud_score": 0.23,
  "risk_level": "LOW",
  "reasons": []
}
```

**Response Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `is_fraud` | boolean | True if transaction is predicted as fraud |
| `fraud_score` | float | Anomaly score from Isolation Forest (higher = more anomalous, **not** a calibrated probability) |
| `risk_level` | string | Risk bucket: "LOW", "MEDIUM", or "HIGH" |
| `reasons` | array | Human-readable reasons for the risk assessment |
| `data_quality_flag` | string \| null | Optional flag: `unseen_customer`, `unseen_merchant`, `unseen_customer_and_merchant`, or `null` when all entities were seen during training |

---

## Example Requests

### Example 1: Normal Transaction (Low Risk)

**Request:**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer": "C1093826151",
    "age": "4",
    "gender": "M",
    "zipcodeOri": "28007",
    "merchant": "M348934600",
    "zipMerchant": "28007",
    "category": "es_transportation",
    "amount": 35.0,
    "step": 100
  }'
```

**Response:**
```json
{
  "is_fraud": false,
  "fraud_score": 0.18,
  "risk_level": "LOW",
  "reasons": []
}
```

---

### Example 2: High-Risk Transaction (Night, High Amount)

**Request:**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer": "C1093826151",
    "age": "4",
    "gender": "M",
    "zipcodeOri": "28007",
    "merchant": "M348934600",
    "zipMerchant": "90210",
    "category": "es_leisure",
    "amount": 5000.0,
    "step": 3
  }'
```

**Response:**
```json
{
  "is_fraud": true,
  "fraud_score": 0.92,
  "risk_level": "HIGH",
  "reasons": [
    "Transaction amount deviates significantly from customer's historical average",
    "Transaction occurred during night hours",
    "Customer and merchant zipcodes differ, indicating a cross-location transaction",
    "Overall anomaly score is in the HIGH risk band"
  ]
}
```

---

### Example 3: Postman Collection

**Import into Postman:**

1. Method: `POST`
2. URL: `http://127.0.0.1:8000/predict`
3. Headers:
   - `Content-Type: application/json`
4. Body (raw JSON):
```json
{
  "customer": "C1093826151",
  "age": "4",
  "gender": "M",
  "zipcodeOri": "28007",
  "merchant": "M348934600",
  "zipMerchant": "28007",
  "category": "es_transportation",
  "amount": 50.0,
  "step": 100
}
```

---

## Feature Engineering

The API computes the same 17 features as in the training notebook and uses precomputed aggregations and label encoders loaded at startup.
