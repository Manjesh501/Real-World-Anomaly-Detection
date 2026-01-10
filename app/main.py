from __future__ import annotations

from typing import List
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import TransactionRequest, PredictionResponse
from app.utils_feature_engineering import (
    compute_features,
    load_feature_engineering_artifacts,
    load_isolation_forest_model,
)


app = FastAPI(
    title="BankSim Fraud Detection API",
    description=(
        "Real-time fraud scoring API using a pre-trained Isolation Forest model "
        "and the same feature engineering pipeline used during training."
    ),
    version="1.0.0",
)

# Allow local frontend or tools to call this API easily. Adjust origins in real deployments.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global objects loaded once at startup and reused for all requests
feature_artifacts = load_feature_engineering_artifacts()
iso_model = load_isolation_forest_model()

logger = logging.getLogger("fraud_api")
logger.setLevel(logging.INFO)

# File-based logging for predictions
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / "predictions.log"

file_handler = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=3)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.propagate = False


def _seen_customer(customer_id: str) -> bool:
    agg = feature_artifacts.aggregations
    if customer_id in agg.customer_txn_count:
        return True
    quoted = f"'{customer_id}'"
    return quoted in agg.customer_txn_count


def _seen_merchant(merchant_id: str) -> bool:
    agg = feature_artifacts.aggregations
    if merchant_id in agg.merchant_txn_count:
        return True
    quoted = f"'{merchant_id}'"
    return quoted in agg.merchant_txn_count


def _risk_from_score(score: float) -> str:
    """Map anomaly score to qualitative risk level.

    The Isolation Forest was trained with contamination≈1.2%. In practice, anomaly
    scores are not guaranteed to fall into a fixed numeric range across datasets,
    so we use simple heuristic thresholds:

    - LOW:    score < 0.5
    - MEDIUM: 0.5 ≤ score < 0.8
    - HIGH:   score ≥ 0.8

    These buckets can be revisited later without retraining the model.
    """

    if score >= 0.8:
        return "HIGH"
    if score >= 0.5:
        return "MEDIUM"
    return "LOW"


def _build_reasons(flags: dict, risk_level: str, is_fraud: bool) -> List[str]:
    """Convert boolean flags into human-readable reasons.

    This function keeps the explanation logic separate from the ML model so that
    it can evolve without changing the underlying Isolation Forest.
    """

    reasons: List[str] = []

    if flags.get("high_amount_deviation"):
        reasons.append(
            "Transaction amount deviates significantly from customer's historical average"
        )
    if flags.get("night_transaction"):
        reasons.append("Transaction occurred during night hours")
    if flags.get("weekend_transaction"):
        reasons.append("Transaction occurred on a weekend")
    if flags.get("cross_location"):
        reasons.append(
            "Customer and merchant zipcodes differ, indicating a cross-location transaction"
        )
    if flags.get("high_velocity_customer"):
        reasons.append("Customer shows unusually high transaction velocity over recent steps")

    # If we did not trigger any specific rule but the model still thinks it's
    # high risk, add a generic explanation.
    if not reasons and is_fraud:
        reasons.append("Model identified this transaction as anomalous relative to historical patterns")

    # Optionally enrich explanations based on risk level
    if risk_level == "HIGH" and is_fraud:
        reasons.append("Overall anomaly score is in the HIGH risk band")
    elif risk_level == "MEDIUM" and is_fraud:
        reasons.append("Overall anomaly score is in the MEDIUM risk band")

    return reasons


@app.get("/health", summary="Health check")
async def health_check() -> dict:
    """Simple health endpoint to verify that the API is running."""

    return {"status": "ok"}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Score a single transaction for fraud risk",
)
async def predict_transaction(payload: TransactionRequest) -> PredictionResponse:
    """Score a single transaction using the Isolation Forest model.

    Steps:
    1. Convert raw transaction into the 17 engineered features used at training time.
    2. Compute Isolation Forest decision_function to obtain an anomaly score.
    3. Convert the model output to a binary fraud label.
    4. Map the score to a qualitative risk level.
    5. Generate a list of human-readable reasons.
    """

    # 1) Compute engineered features from raw transaction
    features, flags = compute_features(payload.dict(), feature_artifacts)

    # 2) Model inference
    # decision_function returns a score where lower values are more anomalous.
    # We convert this to a positive "fraud_score" where higher = more anomalous.
    decision_scores = iso_model.decision_function(features)
    raw_score = float(decision_scores[0])
    fraud_score = float(-raw_score)

    # IsolationForest.predict: -1 = anomaly, 1 = normal
    y_pred_raw = iso_model.predict(features)
    is_fraud = bool(y_pred_raw[0] == -1)

    # 3) Determine qualitative risk level using model decision,
    #    anomaly score, and number of triggered signals.
    num_signals = sum(1 for v in flags.values() if v)

    if is_fraud:
        # For transactions the model flags as fraud, we never return LOW risk.
        # HIGH risk if either the anomaly score is above the fraud threshold
        # (score >= 0 means decision_function <= 0) or if multiple independent
        # signals are triggered.
        HIGH_SCORE_THRESHOLD = 0.0
        if fraud_score >= HIGH_SCORE_THRESHOLD or num_signals >= 3:
            risk_level = "HIGH"
        else:
            risk_level = "MEDIUM"
    else:
        # For transactions the model considers normal, we keep LOW by default
        # and upgrade to MEDIUM only when the anomaly score is slightly positive.
        LOW_SCORE_THRESHOLD = 0.01
        if fraud_score >= LOW_SCORE_THRESHOLD:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

    # 4) Explanation reasons
    reasons = _build_reasons(flags, risk_level, is_fraud)

    # 5) Data quality check - warn if customer/merchant not in training data
    data_quality_flag = None
    if not _seen_customer(payload.customer):
        data_quality_flag = "unseen_customer"
        reasons.append("WARNING: Customer not in training data (using fallback values)")
    if not _seen_merchant(payload.merchant):
        if data_quality_flag:
            data_quality_flag = "unseen_customer_and_merchant"
        else:
            data_quality_flag = "unseen_merchant"
        reasons.append("WARNING: Merchant not in training data (using fallback values)")
    
    # Log prediction for monitoring
    logger.info(
        "prediction customer=%s merchant=%s amount=%.2f is_fraud=%s risk=%s score=%.4f dq=%s",
        payload.customer,
        payload.merchant,
        payload.amount,
        is_fraud,
        risk_level,
        fraud_score,
        data_quality_flag,
    )
    
    return PredictionResponse(
        is_fraud=is_fraud,
        fraud_score=fraud_score,
        risk_level=risk_level,
        reasons=reasons,
        data_quality_flag=data_quality_flag,
    )
