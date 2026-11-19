import numpy as np
import pytest
from fastapi.testclient import TestClient

import fraudlens.api as api


# Fake model for predictable testing.

class FakeModel:

    classes_ = [0, 1]

    def __init__(self, fraud_probability=0.80):
        self.fraud_probability = fraud_probability

    def predict_proba(self, X):
        assert X.shape == (1, 34)

        p = self.fraud_probability

        return np.array([[1 - p, p]])

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


@pytest.fixture
def valid_payload():

    features = {
        name: 0.0
        for name in api.FEATURE_NAMES
    }

    features["Time"] = 3600.0
    features["Amount"] = 100.0

    return {"features": features}


def test_health(client):

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_valid_prediction(client, valid_payload):

    response = client.post(
        "/predict",
        json=valid_payload,
    )

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] == 1
    assert result["label"] == "Fraud"
    assert result["threshold"] == 0.35
    assert result["fraud_probability"] == 0.80


def test_missing_features(client):

    response = client.post(
        "/predict",
        json={
            "features": {
                "Time": 100,
                "Amount": 50,
            }
        },
    )

    assert response.status_code == 422


def test_negative_amount(client, valid_payload):

    valid_payload["features"]["Amount"] = -100

    response = client.post(
        "/predict",
        json=valid_payload,
    )

    assert response.status_code == 422


def test_fraud_prediction(client, valid_payload):

    response = client.post(
        "/predict",
        json=valid_payload,
    )

    result = response.json()

    assert result["prediction"] == 1
    assert result["label"] == "Fraud"


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
    )

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] == 0
    assert result["label"] == "Legitimate"

