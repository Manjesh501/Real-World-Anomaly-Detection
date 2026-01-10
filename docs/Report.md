# Real-World Anomaly Detection with API – Technical Report

## Dataset Choice

Instead of the hypothetical `transactions.csv` described in `requirement.md`, I use the **BankSim** dataset, a public synthetic dataset of card transactions:

- ~594,643 transactions
- 10 columns:  
  `step`, `customer`, `age`, `gender`, `zipcodeOri`, `merchant`, `zipMerchant`, `category`, `amount`, `fraud`
- Fraud rate ≈ **1.2%** (highly imbalanced, similar to real card fraud).

This differs from the assignment’s example schema (no `card_number`, `fraud_type`, or `distance_from_home` in the raw data). To stay aligned with the intent, I engineered features that approximate:

- Temporal context (hour, day of week, weekend/night flags).
- Behavioral context (customer/merchant history, transaction velocity).
- Spatial context (same vs different zip codes as a proxy for distance).

All downstream models and the API operate on the **feature-engineered BankSim dataset**, not on raw assignment columns.

---

## 2. Feature Engineering – Choices and Rationale

Implemented in: `02_feature_engineering.ipynb`  
Used at inference time via: `app/utils_feature_engineering.py`

Starting from the BankSim schema, I engineered a set of **17 features** designed for anomaly detection:

### 2.1 Temporal Features

From `step` (hours since start):

- `hour` (0–23): fraud behavior often peaks at certain hours (e.g., late night).
- `day_of_week` (0–6): weekday vs weekend patterns.
- `is_night` (binary): 1 if hour < 6 or > 22; captures riskier night activity.
- `is_weekend` (binary): 1 for Saturday/Sunday; behavior differs on weekends.

**Why**: Fraud often clusters in specific time windows (night, weekends, bursts).

### 2.2 Spatial Features

From `zipcodeOri` (customer) and `zipMerchant` (merchant):

- `same_zip` (0/1): whether customer and merchant are in the same zip code.
- `distance_proxy` (0/1): 0 if same zip, 1 if different (proxy for distance).

**Why**: Cross-location transactions are more likely to be anomalous, especially when they don’t match the customer’s historical pattern.

### 2.3 Customer-Level Behavioral Features

Using historical BankSim data:

- `customer_txn_count`: total transactions per customer.
- `customer_avg_amount`: average spend per customer.
- `amount_deviation_abs`: absolute deviation from customer’s average.
- `txn_count_last_24`: median recent transaction count (rolling 24-hour window proxy).

**Why**:
- Distinguish new/inactive vs heavy users.
- Large deviations from a customer’s normal spend are often suspicious.
- Sudden bursts in transaction volume can indicate fraud attacks.

### 2.4 Merchant-Level Behavioral Features

- `merchant_txn_count`: total transactions per merchant.
- `cust_merchant_txn_count`: transactions between a specific customer–merchant pair.

**Why**:
- Low-volume or new merchants can be riskier.
- Unusual customer–merchant pairs (no previous history) can be suspicious.

### 2.5 Encoded Categorical Features

- `category_enc`: label-encoded merchant category.
- `gender_enc`: label-encoded gender.
- `age_enc`: label-encoded age bucket.

**Why**: Categorical signals are important context, but anomaly models require numeric inputs. Encodings are learned from the training data and used consistently in the API. Unseen categories are mapped to an “unknown” index at inference time.

### 2.6 Amount Transformations

- `amount`: raw transaction amount.
- `amount_log`: log-transformed amount (`log1p`).

**Why**: Amounts are heavily skewed; log scaling helps anomaly models treat extreme values more smoothly while preserving outliers.

### 2.7 Design Constraints

- **No label leakage**: all engineered features are computed using historical behavior, not future labels.
- **Reproducible in production**: the API’s `compute_features` function reproduces the same transformations from raw JSON.
- **Graceful handling of unseen entities**: for new customers/merchants, the system falls back to global statistics and sets a `data_quality_flag` in the API response.

---

## 3. Models Trained and Evaluation

Implemented in:

- `03_isolation_forest.ipynb`
- `04_one_class_svm.ipynb`
- `05_autoencoder.ipynb`

All models are trained on the same feature matrix (`X_features.csv`) and labels (`y_labels.csv`).

### 3.1 Isolation Forest

**Notebook**: `03_isolation_forest.ipynb`  
**Library**: `sklearn.ensemble.IsolationForest`

Key configuration:
- `n_estimators = 200`
- `contamination ≈ 0.012` (matching the fraud rate)
- Trained on the engineered 17-dimensional feature space.

Evaluation:
- Train/test split with stratification on fraud label.
- Convert `IsolationForest.predict` output (`-1` = anomaly, `1` = normal) into fraud labels (`1` = fraud).
- Metrics:
  - Precision
  - Recall
  - F1
  - ROC-AUC (using negative decision_function as anomaly score)
- The metrics show **high ROC-AUC and strong recall**, making it a strong candidate for production.

### 3.2 One-Class SVM (OCSVM)

**Notebook**: `04_one_class_svm.ipynb`  
**Library**: `sklearn.svm.OneClassSVM`

Key configuration:
- `kernel = "rbf"`
- `nu = 0.012`
- `gamma = "scale"`
- Features are standardized with `StandardScaler`.

Evaluation:
- Similar train/test split.
- Convert OCSVM outputs (`-1`/`1`) to fraud labels and compute the same metrics as for Isolation Forest.
- OCSVM works but is **more sensitive to scaling and hyperparameters**, and in practice it underperforms Isolation Forest on BankSim in terms of ROC-AUC and recall.

### 3.3 Autoencoder

**Notebook**: `05_autoencoder.ipynb`  
**Libraries**: TensorFlow / Keras

Architecture:
- Train only on **normal transactions** (`y == 0`).
- Input: scaled features.
- Encoder: 32 → 16 units (ReLU).
- Decoder: 32 units → back to input dimension.
- Loss: mean squared error (MSE).
- Early stopping on validation loss.

Evaluation:
- Compute reconstruction error for all samples.
- Set anomaly threshold as the 95th percentile of reconstruction error on normal data.
- Label transaction as fraud if error > threshold.
- Compute Precision, Recall, F1, ROC-AUC against true fraud labels.

The autoencoder achieves competitive ROC-AUC but is **more complex to train and deploy**, and less interpretable than Isolation Forest.

### 3.4 Model Selection Justification

Based on experiments across the three approaches:

- **Isolation Forest**:
  - Strong ROC-AUC and recall.
  - Robust to extreme class imbalance.
  - Fast inference (simple tree ensemble).
  - Easier to explain at a high level ("isolates anomalies").

- **OCSVM**:
  - Works as an anomaly detector but more fragile:
    - Sensitive to scaling and parameter choices (`nu`, `gamma`).
    - Generally weaker performance on this dataset.

- **Autoencoder**:
  - Good at modeling complex patterns.
  - More training complexity and less straightforward to interpret.
  - Requires a deep learning stack in production.

**Decision**: Isolation Forest is used in the production API as the primary model. OCSVM and the autoencoder are kept as documented experiments and baselines to discuss trade-offs (accuracy vs complexity vs interpretability).

### 3.5 Quantitative Comparison

On the held-out test set, the three models achieved the following metrics (fraud = positive class):

| Model            | Precision (%) | Recall (%) | F1 (%) | ROC-AUC (%) |
|------------------|--------------:|-----------:|-------:|------------:|
| Isolation Forest |        62.51  |     62.99  |  62.74 |       98.33 |
| One-Class SVM    |        43.48  |     43.33  |  43.41 |       87.72 |
| Autoencoder      |        17.54  |     86.75  |  29.17 |       96.84 |

Isolation Forest gives the best balance of ROC-AUC and F1 while staying simple to deploy and explain, which is why it is the model exposed through the FastAPI endpoint.

---

## 4. FastAPI Inference API

Implemented in: `app/main.py`, `app/schemas.py`, `app/utils_feature_engineering.py`  
Documented in: `docs/README_API.md`

### 4.1 Endpoint Design

- `POST /predict`:
  - Request body: raw transaction JSON with fields matching BankSim:
    - `customer`, `age`, `gender`, `zipcodeOri`, `merchant`, `zipMerchant`, `category`, `amount`, `step`
  - Response:
    - `is_fraud: bool`
    - `fraud_score: float` (Isolation Forest anomaly score; higher = more anomalous, **not a calibrated probability**)
    - `risk_level: "LOW" | "MEDIUM" | "HIGH"` (rule-based mapping)
    - `reasons: List[str]` (human-readable explanations)
    - `data_quality_flag: Optional[str]` (e.g., `unseen_customer`, `unseen_merchant`, or `unseen_customer_and_merchant`)

### 4.2 Request Validation and Error Handling

- Request schema defined via Pydantic (`TransactionRequest`):
  - Types and constraints (e.g., non-negative `amount` and `step`).
  - Required fields enforced by FastAPI.
- Invalid payloads (missing or wrong-type fields) return a 422 error automatically.
- Model artifacts (feature engineering stats and Isolation Forest model) are loaded once at startup; if missing, the application fails fast instead of producing partial results.

### 4.3 Feature Computation at Inference Time

- The API uses `compute_features` to recreate the exact 17-feature vector used during training.
- It handles:
  - Temporal features from `step`.
  - Behavioral features from precomputed aggregations (`banksim_feature_engineered.csv`).
  - Categorical encodings using label mappings.
  - Graceful fallbacks for unseen customers/merchants and unknown categories.

Special handling for ID formats:
- The BankSim CSV includes identifiers with embedded single quotes (e.g., `'C1093826151'`).
- Inference receives IDs without quotes (e.g., `C1093826151`).
- The code normalizes and checks both forms so that known customers/merchants are correctly recognized and do not get incorrectly flagged as unseen.

### 4.4 Risk Level Logic and Reasoning

Isolation Forest returns:
- Decision function value (`decision_function`): higher = more normal.
- Binary decision (`predict`): `-1` = anomaly, `1` = normal.

In the API:
- `fraud_score = -decision_function` (higher = more anomalous).
- Binary prediction: `is_fraud = (predict == -1)`.

Risk level is assigned by a deterministic rule:
- If `is_fraud == true`:
  - `HIGH` if `fraud_score` exceeds a threshold or multiple risk signals are active.
  - Otherwise `MEDIUM`.
- If `is_fraud == false`:
  - `MEDIUM` for slightly positive anomaly scores.
  - Otherwise `LOW`.

Reasons are generated based on flags from the feature computation, such as:
- “Transaction amount deviates significantly from customer’s historical average”
- “Transaction occurred during night hours”
- “Customer and merchant zipcodes differ, indicating a cross-location transaction”
- Generic band-based message for high or medium scores.

### 4.5 Data Quality Flags

When a customer or merchant has no history in the training data:
- The system uses global fallback statistics for feature computation.
- A `data_quality_flag` is set (`unseen_customer`, `unseen_merchant`, or both).
- Corresponding warnings are added to `reasons`.

This makes the model’s limitations explicit in the API response, which is important during a review.

---

## 5. Logging, Monitoring, and Testing

### 5.1 Logging for Predictions

- Logging is implemented using Python’s `logging` module with a rotating file handler.
- Logs are written to:
  - `logs/predictions.log`
- Each prediction logs:
  - Timestamp
  - Customer ID
  - Merchant ID
  - Amount
  - `is_fraud`
  - `risk_level`
  - `fraud_score`
  - `data_quality_flag`


### 5.2 Unit Tests for the API

Implemented in:

- `tests/test_api_setup.py`
  - Verifies:
    - Modules import correctly.
    - Feature engineering artifacts are loaded.
    - The Isolation Forest model is loaded and can score transactions.
    - Basic normal / high-risk / unseen-customer scenarios work.
- `tests/test_api_routes.py`
  - Uses FastAPI `TestClient` to exercise HTTP endpoints:
    - `/health` returns status 200 and `{"status": "ok"}`.
    - `/predict` with a valid normal payload returns a consistent schema and types.
    - `/predict` high-risk case returns `is_fraud = true` and non-LOW risk.
    - `/predict` with invalid payload (missing required field) returns 422.

All tests pass, giving confidence in both the core logic and the HTTP layer.

---

## 6. Edge Cases and Production Considerations

### 6.1 Edge Cases Covered

Examples tested and discussed:

- **High-risk transaction**:
  - Night-time, high amount, different zip codes, and inconsistent with customer history.
  - Model flags it as fraud, with `HIGH` risk and multiple reasons.

- **Unseen customer/merchant**:
  - First-time customer/merchant combinations not present in training data.
  - Features fall back to global statistics.
  - `data_quality_flag` and warnings in `reasons` clearly indicate this.

- **Normal baseline transactions**:
  - Typical daytime, local zip, amount close to customer’s historical average.
  - Model returns `is_fraud = false`, `LOW` risk, and usually an empty reasons list.

### 6.2 Deployment Considerations

- The API is stateless and loads artifacts at startup:
  - `banksim_feature_engineered.csv` from `data/processed`.
  - `isolation_forest_model.pkl` from `models`.
- The project structure separates:
  - `app/` for API code.
  - `data/` for raw and processed data.
  - `models/` for artifacts.
  - `notebooks/` for experiments.
  - `tests/` for validation scripts.

### 6.3 Summary Statistics

A few key numbers that summarize the project:

- Dataset: 594,643 transactions (BankSim), fraud rate ≈ 1.21% (7,200 frauds).
- Features engineered: 17 numeric features tailored for anomaly detection.
- Models built: Isolation Forest, One-Class SVM, and Autoencoder.
- Train/test split: 80% training, 20% test, stratified by fraud label.
- Best model (Isolation Forest) test performance:
  - Precision ≈ 62.5%
  - Recall ≈ 63.0%
  - F1 ≈ 62.7%
  - ROC-AUC ≈ 98.3%

These figures are consistent with the EDA and model notebooks and are the basis for selecting Isolation Forest as the production model.
---
