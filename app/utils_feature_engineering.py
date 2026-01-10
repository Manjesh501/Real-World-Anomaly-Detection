from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple, Any

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "processed" / "banksim_feature_engineered.csv"
MODEL_PATH = BASE_DIR.parent / "models" / "isolation_forest_model.pkl"

# BankSim feature engineering used a synthetic start time for step → timestamp conversion.
START_TIME = datetime(2024, 1, 1)


@dataclass
class LabelEncodingArtifacts:
    category_mapping: Dict[str, int]
    gender_mapping: Dict[str, int]
    age_mapping: Dict[str, int]


@dataclass
class AggregationArtifacts:
    customer_txn_count: Dict[str, int]
    customer_avg_amount: Dict[str, float]
    cust_merchant_txn_count: Dict[Tuple[str, str], int]
    merchant_txn_count: Dict[str, int]
    customer_txn_count_last_24_median: Dict[str, float]
    global_amount_mean: float
    global_amount_std: float
    global_txn_count_last_24_median: float


@dataclass
class FeatureEngineeringArtifacts:
    label_encoders: LabelEncodingArtifacts
    aggregations: AggregationArtifacts


def _build_label_mapping(series: pd.Series) -> Dict[str, int]:
    """Mimic sklearn's LabelEncoder (sorted unique values → integer codes).

    Unknown values at inference time are mapped to a dedicated "unknown" index
    at the end of the known classes.
    """

    unique_values = sorted(series.astype(str).unique())
    mapping = {val: idx for idx, val in enumerate(unique_values)}
    return mapping


def load_feature_engineering_artifacts() -> FeatureEngineeringArtifacts:
    """Load precomputed statistics from the training dataset.

    This function reproduces the logic from 02_feature_engineering.ipynb by
    reading `banksim_feature_engineered.csv` and computing:
    - Label encodings for category, gender, age
    - Per-customer and per-merchant aggregations
    - Typical values for txn_count_last_24 and amount deviation thresholds

    These artifacts are used to compute features for NEW transactions in a
    way that is consistent with training, while handling unseen customers/
    merchants/categories gracefully.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Expected engineered data at {DATA_PATH}. Make sure 02_feature_engineering.ipynb was run."
        )

    df = pd.read_csv(DATA_PATH)

    # Label encoders (mimic LabelEncoder behavior)
    category_mapping = _build_label_mapping(df["category"])
    gender_mapping = _build_label_mapping(df["gender"])
    age_mapping = _build_label_mapping(df["age"])

    label_artifacts = LabelEncodingArtifacts(
        category_mapping=category_mapping,
        gender_mapping=gender_mapping,
        age_mapping=age_mapping,
    )

    # Aggregations used during feature engineering
    customer_txn_count = df.groupby("customer").size().to_dict()
    customer_avg_amount = df.groupby("customer")["amount"].mean().to_dict()
    cust_merchant_txn_count = (
        df.groupby(["customer", "merchant"]).size().to_dict()
    )
    merchant_txn_count = df.groupby("merchant").size().to_dict()

    # txn_count_last_24 was computed per-row in the original notebook.
    # For online inference, we cannot reconstruct a rolling window, so we use
    # the per-customer median of historical txn_count_last_24 as a stable
    # proxy, falling back to the global median for unseen customers.
    if "txn_count_last_24" not in df.columns:
        raise ValueError(
            "Expected 'txn_count_last_24' column in banksim_feature_engineered.csv."
        )

    per_customer_median = (
        df.groupby("customer")["txn_count_last_24"].median().to_dict()
    )

    global_amount_mean = float(df["amount"].mean())
    global_amount_std = float(df["amount"].std())
    global_txn_count_last_24_median = float(df["txn_count_last_24"].median())

    agg_artifacts = AggregationArtifacts(
        customer_txn_count=customer_txn_count,
        customer_avg_amount=customer_avg_amount,
        cust_merchant_txn_count=cust_merchant_txn_count,
        merchant_txn_count=merchant_txn_count,
        customer_txn_count_last_24_median=per_customer_median,
        global_amount_mean=global_amount_mean,
        global_amount_std=global_amount_std,
        global_txn_count_last_24_median=global_txn_count_last_24_median,
    )

    return FeatureEngineeringArtifacts(
        label_encoders=label_artifacts,
        aggregations=agg_artifacts,
    )


def _encode_with_unknown(mapping: Dict[str, int], value: str) -> int:
    """Encode a categorical value using a mapping.

    If the value was never seen during training, map it to a dedicated
    "unknown" index (len(mapping)).
    """

    if value in mapping:
        return mapping[value]
    return len(mapping)


def compute_features(
    raw: Dict[str, Any],
    artifacts: FeatureEngineeringArtifacts,
) -> Tuple[np.ndarray, Dict[str, bool]]:
    """Compute the 17-feature vector used by Isolation Forest.

    Parameters
    ----------
    raw: dict
        Parsed transaction payload (already validated by Pydantic).
    artifacts: FeatureEngineeringArtifacts
        Precomputed label encoders and aggregations.

    Returns
    -------
    features: np.ndarray of shape (1, 17)
        Feature vector in the exact order used during training.
    flags: dict
        Boolean flags used later to generate human-readable reasons.
    """

    le = artifacts.label_encoders
    agg = artifacts.aggregations

    # Normalize identifiers to match training artifacts (handle possible single quotes)
    def _match_key(key: str, mapping: Dict[str, Any]) -> str:
        if key in mapping:
            return key
        quoted = f"'{key}'"
        if quoted in mapping:
            return quoted
        return key

    def _match_pair_key(cust: str, merch: str, mapping: Dict[Tuple[str, str], Any]) -> Tuple[str, str]:
        if (cust, merch) in mapping:
            return cust, merch
        quoted_pair = (f"'{cust}'", f"'{merch}'")
        if quoted_pair in mapping:
            return quoted_pair
        return cust, merch

    customer_raw = str(raw["customer"])
    age = str(raw["age"])
    gender = str(raw["gender"])
    zipcode_ori = str(raw["zipcodeOri"])
    merchant_raw = str(raw["merchant"])
    zip_merchant = str(raw["zipMerchant"])
    category_raw = str(raw["category"])
    amount = float(raw["amount"])
    step = int(raw["step"])

    # Use keys that best match training mappings
    customer = _match_key(customer_raw, agg.customer_txn_count)
    merchant = _match_key(merchant_raw, agg.merchant_txn_count)
    cust_key, merch_key_for_pair = _match_pair_key(customer, merchant, agg.cust_merchant_txn_count)

    # For label encoders, try both raw and quoted values
    def _norm_for_le(val: str, mapping: Dict[str, int]) -> str:
        if val in mapping:
            return val
        quoted = f"'{val}'"
        if quoted in mapping:
            return quoted
        return val

    category = _norm_for_le(category_raw, le.category_mapping)
    gender_val = _norm_for_le(gender, le.gender_mapping)
    age_val = _norm_for_le(age, le.age_mapping)

    # --- Temporal features ---
    timestamp = START_TIME + timedelta(hours=step)
    hour = timestamp.hour
    day_of_week = timestamp.weekday()  # 0=Mon, 6=Sun
    month = timestamp.month

    is_night = int(hour < 6 or hour > 22)
    is_weekend = int(day_of_week >= 5)

    # --- Spatial features ---
    same_zip = int(zipcode_ori == zip_merchant)
    distance_proxy = 0 if same_zip == 1 else 1

    # --- Aggregation-based behavioral features ---
    customer_txn_count = agg.customer_txn_count.get(customer, 1)
    customer_avg_amount = agg.customer_avg_amount.get(
        customer, agg.global_amount_mean
    )

    amount_deviation = amount - customer_avg_amount
    amount_deviation_abs = abs(amount_deviation)

    merchant_txn_count = agg.merchant_txn_count.get(merchant, 0)
    cust_merchant_txn_count = agg.cust_merchant_txn_count.get(
        (cust_key, merch_key_for_pair), 1
    )

    # For txn_count_last_24 we use median historical window per customer.
    txn_count_last_24 = agg.customer_txn_count_last_24_median.get(
        customer, agg.global_txn_count_last_24_median
    )

    # --- Encoded categories ---
    category_enc = _encode_with_unknown(le.category_mapping, category)
    gender_enc = _encode_with_unknown(le.gender_mapping, gender_val)
    age_enc = _encode_with_unknown(le.age_mapping, age_val)

    # --- Amount transformations ---
    amount_log = float(np.log1p(amount))

    # Order MUST match feature_cols in 02_feature_engineering.ipynb
    feature_vector = np.array(
        [
            amount,
            amount_log,
            amount_deviation_abs,
            hour,
            day_of_week,
            is_night,
            is_weekend,
            customer_txn_count,
            customer_avg_amount,
            txn_count_last_24,
            cust_merchant_txn_count,
            same_zip,
            distance_proxy,
            category_enc,
            gender_enc,
            age_enc,
            merchant_txn_count,
        ],
        dtype=float,
    ).reshape(1, -1)

    # Flags for explainability
    flags = {
        "high_amount_deviation": amount_deviation_abs
        > (agg.global_amount_std * 2.0),
        "night_transaction": bool(is_night),
        "weekend_transaction": bool(is_weekend),
        "cross_location": bool(distance_proxy == 1),
        "high_velocity_customer": txn_count_last_24
        > (agg.global_txn_count_last_24_median * 2.0),
    }

    return feature_vector, flags


def load_isolation_forest_model():
    """Load the trained Isolation Forest model from disk.

    The model is loaded once at application startup and reused for all
    incoming requests.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Isolation Forest model not found at {MODEL_PATH}. Ensure 03_isolation_forest.ipynb was executed."
        )

    model = joblib.load(MODEL_PATH)
    return model
