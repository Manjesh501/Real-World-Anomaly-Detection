# 🛡️ Fraud Detection API

**Anomaly detection system for financial fraud prevention**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Features](#features)
- [Model Performance](#model-performance)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Design Decisions](#design-decisions)
---

## 🎯 Overview

This project implements a **real-time fraud detection system** using unsupervised anomaly detection on transactional data. Built for production environments, it handles highly imbalanced datasets where fraudulent transactions represent only ~1-2% of total activity.

### Key Highlights

✅ **Unsupervised Learning** – No labeled fraud data required for training  
✅ **Real-time Inference** – <100ms response time via FastAPI  
✅ **Explainable Results** – Risk scores with human-readable reasoning  
✅ **Production-ready** – Robust feature engineering and error handling  
✅ **Model Comparison** – Evaluated Isolation Forest, One-Class SVM, and Autoencoder

---

## 🧩 Problem Statement

### Challenge

Financial fraud datasets suffer from **extreme class imbalance**, with fraudulent transactions typically representing only 1-2% of total activity. Traditional supervised models struggle with:

- Insufficient fraud examples for training
- High false positive rates
- Poor generalization to novel fraud patterns

### Solution

An **anomaly detection approach** that:

1. **Learns normal transaction behavior** from legitimate transactions
2. **Flags statistical outliers** as potentially fraudulent
3. **Provides interpretable outputs** for human review
4. **Operates in real-time** with sub-100ms latency

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Source** | BankSim (Synthetic Payment Simulator) |
| **Records** | ~594,000 transactions |
| **Fraud Rate** | 1.21% (7,200 fraudulent) |
| **Type** | Synthetic, privacy-safe transactional data |
| **Features** | Customer, merchant, amount, location, time, category |

### Data Quality

✅ No missing values  
✅ No duplicates  
✅ Balanced temporal distribution  
✅ Realistic fraud patterns

---

## 🔧 Features

### Feature Engineering Pipeline

Engineered **17 transaction-time features** across four categories:

#### 1. **Amount Features**
- `amount` – Raw transaction value
- `amount_log` – Log-transformed amount
- `amount_deviation_abs` – Deviation from customer's historical average

#### 2. **Temporal Features**
- `hour` – Hour of transaction
- `day_of_week` – Day of week (0=Monday)
- `is_night` – Night transaction flag (10PM-6AM)
- `is_weekend` – Weekend transaction flag

#### 3. **Behavioral Features**
- `customer_txn_count` – Customer's total transaction count
- `customer_avg_amount` – Customer's average transaction amount
- `txn_count_last_24` – Customer's transactions in last 24 hours
- `cust_merchant_txn_count` – Customer-merchant interaction frequency
- `merchant_txn_count` – Merchant's total transaction count

#### 4. **Location Features**
- `same_zip` – Customer and merchant in same ZIP code
- `distance_proxy` – Cross-location transaction indicator

#### 5. **Categorical Encodings**
- `category_enc` – Merchant category (label encoded)
- `gender_enc` – Customer gender (label encoded)
- `age_enc` – Customer age bracket (label encoded)

✅ **No data leakage** – Fraud label never used in features  
✅ **Real-time compatible** – All features computable at transaction time

---

## 📈 Model Performance

### Models Evaluated

| Model | ROC-AUC | Precision | Recall | F1-Score | Selection |
|-------|---------|-----------|--------|----------|----------|
| **Isolation Forest** | **98.33%** | **62.5%** | **63.0%** | **62.7%** | ✅ **Deployed** |
| One-Class SVM | 87.72% | 43.5% | 43.3% | 43.4% | Baseline |
| Autoencoder | 96.84% | 17.5% | 86.8% | 29.2% | High-recall option |

### Why Isolation Forest?

**Isolation Forest** was selected as the production model due to:

✅ **Best balanced performance** – Strong precision/recall trade-off  
✅ **High ROC-AUC (98.33%)** – Excellent anomaly separation  
✅ **Low latency** – Fast inference suitable for real-time systems  
✅ **Interpretability** – Anomaly scores provide clear decision boundaries  
✅ **Scalability** – Efficient training and inference on large datasets

### Performance Interpretation

- **ROC-AUC: 98.33%** → Model effectively separates fraud from legitimate transactions
- **Precision: 62.5%** → ~62% of flagged transactions are actual fraud
- **Recall: 63.0%** → ~63% of all fraud cases are detected
- **Trade-off** → Balanced for environments where both false positives and missed fraud are costly

---

## 🚀 API Documentation

### Endpoint

```http
POST /predict
Content-Type: application/json
```

### Request Schema

```json
{
  "customer": "C1093826151",
  "age": "4",
  "gender": "M",
  "zipcodeOri": "28007",
  "merchant": "M348934600",
  "zipMerchant": "90210",
  "category": "es_leisure",
  "amount": 5000.0,
  "step": 3
}
```

### Response Schema

```json
{
  "is_fraud": true,
  "fraud_score": 0.074,
  "risk_level": "HIGH",
  "reasons": [
    "Transaction amount deviates significantly from customer's historical average",
    "Transaction occurred during night hours",
    "Customer and merchant zipcodes differ, indicating a cross-location transaction"
  ]
}
```

### Risk Levels

| Level | Criteria |
|-------|----------|
| **HIGH** | `is_fraud=true` AND (`fraud_score >= 0.0` OR `num_signals >= 3`) |
| **MEDIUM** | `is_fraud=true` AND other cases, OR `is_fraud=false` AND `fraud_score >= 0.01` |
| **LOW** | `is_fraud=false` AND `fraud_score < 0.01` |

### Interactive Documentation

Once the API is running, visit:
- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

---

## 📁 Project Structure

```plaintext
anamoly/
│
├── app/                          # FastAPI application
│   ├── __init__.py
│   ├── main.py                   # API endpoints and risk logic
│   ├── schemas.py                # Pydantic request/response models
│   └── utils_feature_engineering.py  # Feature computation
│
├── data/
│   ├── raw/                      # Original BankSim data
│   │   └── banksim.csv
│   └── processed/                # Engineered features
│       ├── banksim_feature_engineered.csv
│       ├── X_features.csv
│       └── y_labels.csv
│
├── models/                       # Trained model artifacts
│   ├── isolation_forest_model.pkl
│   ├── one_class_svm_model.pkl
│   └── ocsvm_scaler.pkl
│
├── notebooks/                    # Jupyter notebooks
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_isolation_forest.ipynb
│   ├── 04_one_class_svm.ipynb
│   └── 05_autoencoder.ipynb
│
├── tests/                        # Validation scripts
│   ├── __init__.py
│   └── test_api_setup.py         # Pre-deployment validation
│
│
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🔨 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/Manjesh501/Real-World-Anomaly-Detection
   cd Real-World-Anomaly-Detection
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python tests/test_api_setup.py
   ```

---

## 💻 Usage

### Start the API Server

```bash
uvicorn app.main:app --reload
```

The API will be available at: `http://127.0.0.1:8000`

### Test with cURL

#### Health Check
```bash
curl http://127.0.0.1:8000/health
```

#### Normal Transaction
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

#### High-Risk Transaction
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

### Test with Python

```python
import requests

url = "http://127.0.0.1:8000/predict"
payload = {
    "customer": "C1093826151",
    "age": "4",
    "gender": "M",
    "zipcodeOri": "28007",
    "merchant": "M348934600",
    "zipMerchant": "90210",
    "category": "es_leisure",
    "amount": 5000.0,
    "step": 3
}

response = requests.post(url, json=payload)
print(response.json())
```

---

## 🧠 Design Decisions

### 1. **Anomaly Detection Over Supervised Learning**

**Rationale**: Extreme class imbalance (1.21% fraud rate) makes supervised learning prone to overfitting on the minority class. Anomaly detection learns from the majority class (legitimate transactions) and identifies statistical outliers.

### 2. **Isolation Forest Over Neural Models**

**Rationale**: While autoencoders achieved higher recall (86.8%), they suffered from extremely low precision (17.5%), resulting in excessive false positives. Isolation Forest provides the best balance for production deployment.

### 3. **Interpretability First**

**Rationale**: Financial fraud detection requires human review of flagged transactions. Providing anomaly scores and rule-based reasons enables efficient investigation and reduces false positive impact.

### 4. **Real-time Feature Engineering**

**Rationale**: All features are computable at transaction time using historical aggregates and current transaction data, enabling sub-100ms inference latency.

### 5. **Rule-based Risk Levels**

**Rationale**: Risk levels (LOW/MEDIUM/HIGH) are derived from model predictions, anomaly scores, and signal counts using deterministic rules, ensuring consistency and auditability.

---