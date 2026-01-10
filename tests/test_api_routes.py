"""Basic API route tests using FastAPI TestClient.

Run this file directly with Python to validate the /health and /predict endpoints.
This is a lightweight alternative to a full pytest suite.
"""

import json

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"


def test_predict_normal_transaction():
    payload = {
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
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data.get("is_fraud"), bool)
    assert isinstance(data.get("fraud_score"), (int, float))
    assert data.get("risk_level") in {"LOW", "MEDIUM", "HIGH"}
    assert isinstance(data.get("reasons"), list)


def test_predict_high_risk_transaction():
    payload = {
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
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data.get("is_fraud") is True
    assert data.get("risk_level") in {"MEDIUM", "HIGH"}


def test_predict_invalid_payload():
    # Missing required field "customer"
    payload = {
        "age": "4",
        "gender": "M",
        "zipcodeOri": "28007",
        "merchant": "M348934600",
        "zipMerchant": "28007",
        "category": "es_transportation",
        "amount": 35.0,
        "step": 100,
    }
    response = client.post("/predict", json=payload)
    # FastAPI / Pydantic should reject this with 422
    assert response.status_code == 422


def main():
    print("Running basic API route tests...")
    test_health()
    print("- /health OK")
    test_predict_normal_transaction()
    print("- /predict normal transaction OK")
    test_predict_high_risk_transaction()
    print("- /predict high-risk transaction OK")
    test_predict_invalid_payload()
    print("- /predict invalid payload handling OK")
    print("ALL API ROUTE TESTS PASSED")


if __name__ == "__main__":
    main()
