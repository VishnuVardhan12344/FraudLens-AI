
import numpy as np
import pytest
from fastapi.testclient import TestClient

import fraudlens.api as api


# -------------------------------------
# 1. TEST AUTHENTICATION CONFIGURATION
# -------------------------------------

TEST_API_KEY = "fraudlens-test-key-only"

AUTH_HEADERS = {
    "X-API-Key": TEST_API_KEY
}


@pytest.fixture(autouse=True)
def configure_test_api_key(monkeypatch):
    monkeypatch.setenv(
        "FRAUDLENS_API_KEY",
        TEST_API_KEY,
    )


# -------------------------------------
# 2. FAKE MACHINE LEARNING MODEL
# -------------------------------------

class FakeModel:

    classes_ = [0, 1]

    def __init__(self, fraud_probability=0.80):
        self.fraud_probability = fraud_probability

    def predict_proba(self, X):
        assert X.shape == (1, 34)

        p = self.fraud_probability

        return np.array([[1 - p, p]])


# -------------------------------------
# 3. TEST CLIENT FIXTURE
# -------------------------------------

@pytest.fixture
def client(monkeypatch):

    # Prevent the API from loading the real model.
    monkeypatch.setattr(
        api,
        "load_model",
        lambda: {
            "model": FakeModel(0.80),
            "threshold": 0.35,
            "feature_names": (
                api.FEATURE_NAMES
                + api.ENGINEERED_FEATURES
            ),
        },
    )

    with TestClient(api.app) as test_client:
        yield test_client


# -------------------------------------
# 4. VALID TRANSACTION FIXTURE
# -------------------------------------

@pytest.fixture
def valid_payload():

    features = {
        name: 0.0
        for name in api.FEATURE_NAMES
    }

    features["Time"] = 3600.0
    features["Amount"] = 100.0

    return {"features": features}


# -------------------------------------
# 5. HEALTH CHECK TEST
# -------------------------------------

def test_health(client):

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


# -------------------------------------
# 6. VALID PREDICTION TEST
# -------------------------------------

def test_valid_prediction(client, valid_payload):

    response = client.post(
        "/predict",
        json=valid_payload,
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] == 1
    assert result["label"] == "Fraud"
    assert result["threshold"] == 0.35
    assert result["fraud_probability"] == 0.80


# -------------------------------------
# 7. MISSING FEATURES TEST
# -------------------------------------

def test_missing_features(client):

    response = client.post(
        "/predict",
        json={
            "features": {
                "Time": 100,
                "Amount": 50,
            }
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 422


# -------------------------------------
# 8. NEGATIVE AMOUNT TEST
# -------------------------------------

def test_negative_amount(client, valid_payload):

    valid_payload["features"]["Amount"] = -100

    response = client.post(
        "/predict",
        json=valid_payload,
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 422


# -------------------------------------
# 9. FRAUD PREDICTION TEST
# -------------------------------------

def test_fraud_prediction(client, valid_payload):

    response = client.post(
        "/predict",
        json=valid_payload,
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] == 1
    assert result["label"] == "Fraud"


# -------------------------------------
# 10. LEGITIMATE PREDICTION TEST
# -------------------------------------

def test_legitimate_prediction(
    client,
    valid_payload,
    monkeypatch,
):

    monkeypatch.setattr(
        api,
        "MODEL_BUNDLE",
        {
            "model": FakeModel(0.10),
            "threshold": 0.35,
            "feature_names": (
                api.FEATURE_NAMES
                + api.ENGINEERED_FEATURES
            ),
        },
    )

    response = client.post(
        "/predict",
        json=valid_payload,
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] == 0
    assert result["label"] == "Legitimate"


# -------------------------------------
# 11. MISSING API KEY TEST
# -------------------------------------

def test_predict_missing_api_key(
    client,
    valid_payload,
):

    response = client.post(
        "/predict",
        json=valid_payload,
    )

    assert response.status_code == 401

    assert response.json()["detail"] == (
        "Missing or invalid API key"
    )


# -------------------------------------
# 12. INVALID API KEY TEST
# -------------------------------------

def test_predict_invalid_api_key(
    client,
    valid_payload,
):

    response = client.post(
        "/predict",
        json=valid_payload,
        headers={
            "X-API-Key": "incorrect-key"
        },
    )

    assert response.status_code == 401

    assert response.json()["detail"] == (
        "Missing or invalid API key"
    )


# -------------------------------------
# 13. UNCONFIGURED API KEY TEST
# -------------------------------------

def test_predict_unconfigured_api_key(
    client,
    valid_payload,
    monkeypatch,
):

    monkeypatch.delenv(
        "FRAUDLENS_API_KEY",
        raising=False,
    )

    response = client.post(
        "/predict",
        json=valid_payload,
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 503

    assert response.json()["detail"] == (
        "API authentication is not configured"
    )
