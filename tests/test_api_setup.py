"""
Test script to validate the FastAPI fraud detection API.

This script tests:
1. Feature engineering artifacts loading
2. Model loading
3. Feature computation
4. End-to-end prediction

Run this before starting the API to ensure everything is configured correctly.
"""

import sys
from pathlib import Path

# Test 1: Import modules
print("=" * 60)
print("TEST 1: Importing modules...")
print("=" * 60)

try:
    from app.schemas import TransactionRequest, PredictionResponse
    from app.utils_feature_engineering import (
        load_feature_engineering_artifacts,
        load_isolation_forest_model,
        compute_features,
    )
    print("✅ All modules imported successfully")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Load feature engineering artifacts
print("\n" + "=" * 60)
print("TEST 2: Loading feature engineering artifacts...")
print("=" * 60)

try:
    artifacts = load_feature_engineering_artifacts()
    print(f"✅ Loaded label encoders:")
    print(f"   - Categories: {len(artifacts.label_encoders.category_mapping)} unique values")
    print(f"   - Genders: {len(artifacts.label_encoders.gender_mapping)} unique values")
    print(f"   - Ages: {len(artifacts.label_encoders.age_mapping)} unique values")
    print(f"✅ Loaded aggregations:")
    print(f"   - Customers: {len(artifacts.aggregations.customer_txn_count)} known customers")
    print(f"   - Merchants: {len(artifacts.aggregations.merchant_txn_count)} known merchants")
    print(f"   - Global avg amount: €{artifacts.aggregations.global_amount_mean:.2f}")
except Exception as e:
    print(f"❌ Failed to load artifacts: {e}")
    sys.exit(1)

# Test 3: Load model
print("\n" + "=" * 60)
print("TEST 3: Loading Isolation Forest model...")
print("=" * 60)

try:
    model = load_isolation_forest_model()
    print(f"✅ Model loaded successfully")
    print(f"   - Model type: {type(model).__name__}")
    print(f"   - n_estimators: {model.n_estimators}")
    print(f"   - contamination: {model.contamination}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    sys.exit(1)

# Test 4: Feature computation (normal transaction)
print("\n" + "=" * 60)
print("TEST 4: Computing features for a normal transaction...")
print("=" * 60)

normal_txn = {
    "customer": "C1093826151",
    "age": "4",
    "gender": "M",
    "zipcodeOri": "28007",
    "merchant": "M348934600",
    "zipMerchant": "28007",
    "category": "es_transportation",
    "amount": 35.0,
    "step": 100,
}

try:
    features, flags = compute_features(normal_txn, artifacts)
    print(f"✅ Feature computation successful")
    print(f"   - Feature shape: {features.shape}")
    print(f"   - Feature vector (first 5): {features[0, :5]}")
    print(f"   - Flags: {flags}")
except Exception as e:
    print(f"❌ Feature computation failed: {e}")
    sys.exit(1)

# Test 5: Model prediction (normal transaction)
print("\n" + "=" * 60)
print("TEST 5: Predicting normal transaction...")
print("=" * 60)

try:
    decision_score = model.decision_function(features)
    prediction = model.predict(features)
    is_fraud = prediction[0] == -1
    fraud_score = float(-decision_score[0])

    print(f"✅ Prediction successful")
    print(f"   - Decision score: {decision_score[0]:.4f}")
    print(f"   - Fraud score: {fraud_score:.4f}")
    print(f"   - Is fraud: {is_fraud}")
    print(f"   - Risk level: {'HIGH' if fraud_score >= 0.8 else 'MEDIUM' if fraud_score >= 0.5 else 'LOW'}")
except Exception as e:
    print(f"❌ Prediction failed: {e}")
    sys.exit(1)

# Test 6: High-risk transaction
print("\n" + "=" * 60)
print("TEST 6: Computing features for a high-risk transaction...")
print("=" * 60)

high_risk_txn = {
    "customer": "C1093826151",
    "age": "4",
    "gender": "M",
    "zipcodeOri": "28007",
    "merchant": "M348934600",
    "zipMerchant": "90210",
    "category": "es_leisure",
    "amount": 5000.0,
    "step": 3,
}

try:
    features, flags = compute_features(high_risk_txn, artifacts)
    decision_score = model.decision_function(features)
    prediction = model.predict(features)
    is_fraud = prediction[0] == -1
    fraud_score = float(-decision_score[0])

    print(f"✅ High-risk transaction processing successful")
    print(f"   - Is fraud: {is_fraud}")
    print(f"   - Fraud score: {fraud_score:.4f}")
    print(f"   - Risk level: {'HIGH' if fraud_score >= 0.8 else 'MEDIUM' if fraud_score >= 0.5 else 'LOW'}")
    print(f"   - Flags triggered:")
    for flag, value in flags.items():
        if value:
            print(f"     ✓ {flag}")
except Exception as e:
    print(f"❌ High-risk transaction failed: {e}")
    sys.exit(1)

# Test 7: Unseen customer (edge case)
print("\n" + "=" * 60)
print("TEST 7: Testing with unseen customer (edge case)...")
print("=" * 60)

unseen_txn = {
    "customer": "C_NEVER_SEEN_BEFORE",
    "age": "999",
    "gender": "X",
    "zipcodeOri": "00000",
    "merchant": "M_UNKNOWN",
    "zipMerchant": "00000",
    "category": "unknown_category",
    "amount": 100.0,
    "step": 500,
}

try:
    features, flags = compute_features(unseen_txn, artifacts)
    decision_score = model.decision_function(features)
    prediction = model.predict(features)
    is_fraud = prediction[0] == -1

    print(f"✅ Unseen customer handled gracefully")
    print(f"   - Feature computation succeeded (uses fallbacks)")
    print(f"   - Is fraud: {is_fraud}")
    print(f"   - No crash on unseen data ✓")
except Exception as e:
    print(f"❌ Unseen customer handling failed: {e}")
    sys.exit(1)

# Final summary
print("\n" + "=" * 60)
print("🎉 ALL TESTS PASSED!")
print("=" * 60)
print("\n✅ The API is ready to start:")
print("   uvicorn main:app --reload")
print("\n✅ Test endpoints:")
print("   curl http://127.0.0.1:8000/health")
print("   curl -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{...}'")
print("\n✅ Interactive docs:")
print("   http://127.0.0.1:8000/docs")
