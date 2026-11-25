
import numpy as np
import pytest

from fastapi.testclient import TestClient

from fraudlens import api

from fraudlens import rate_limiter as limiter_module

from fraudlens.rate_limiter import RateLimiter

from fraudlens.prediction_monitor import PredictionMonitor


TEST_KEY = "fraudlens-database-test-key"

HEADERS = {
    "X-API-Key": TEST_KEY
}


class FakeModel:

    def predict_proba(self, X):

        return np.array([
            [0.10, 0.90]
        ])


@pytest.fixture
def client(monkeypatch):

    monkeypatch.setenv(
        "FRAUDLENS_API_KEY",
        TEST_KEY,
    )

    monkeypatch.setattr(
        api,
        "prediction_monitor",
        PredictionMonitor(),
    )

    monkeypatch.setattr(
        limiter_module,
        "rate_limiter",
        RateLimiter(
            max_requests=60,
            window_seconds=60,
        ),
    )

    monkeypatch.setattr(
        api,
        "initialize_database",
        lambda: True,
    )

    monkeypatch.setattr(
        api,
        "save_prediction",
        lambda **kwargs: True,
    )

    monkeypatch.setattr(
        api,
        "get_prediction_history",
        lambda limit: [
            {
                "request_id": "test-request",
                "fraud_probability": 0.90,
                "prediction": 1,
                "label": "Fraud",
            }
        ],
    )

    monkeypatch.setattr(
        api,
        "get_database_summary",
        lambda: {
            "total_predictions": 1,
            "fraud_predictions": 1,
            "legitimate_predictions": 0,
            "fraud_prediction_rate": 1.0,
            "average_fraud_probability": 0.90,
            "average_latency_ms": 20.0,
        },
    )

    monkeypatch.setattr(
        api,
        "load_model",
        lambda: {
            "model": FakeModel(),
            "threshold": 0.35,
            "feature_names": (
                api.FEATURE_NAMES
                + api.ENGINEERED_FEATURES
            ),
        },
    )

    with TestClient(api.app) as test_client:

        yield test_client


def test_database_status(client):

    response = client.get(
        "/database/status",
        headers=HEADERS,
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


@pytest.mark.parametrize(
    "endpoint",
    [
        "/database/status",
        "/predictions/history",
        "/predictions/summary",
    ],
)
def test_authentication(client, endpoint):

    response = client.get(endpoint)

    assert response.status_code == 401


def test_prediction_storage(client):

    features = {
        name: 0.0
        for name in api.FEATURE_NAMES
    }

    features["Time"] = 1000.0
    features["Amount"] = 120.0

    response = client.post(
        "/predict",
        headers=HEADERS,
        json={"features": features},
    )

    assert response.status_code == 200

    result = response.json()

    assert result["stored_in_database"] is True
    assert result["prediction"] == 1
    assert result["label"] == "Fraud"
    assert result["request_id"]


def test_prediction_history(client):

    response = client.get(
        "/predictions/history",
        headers=HEADERS,
    )

    assert response.status_code == 200
    assert response.json()["count"] == 1


def test_prediction_summary(client):

    response = client.get(
        "/predictions/summary",
        headers=HEADERS,
    )

    assert response.status_code == 200

    assert (
        response.json()["total_predictions"]
        == 1
    )


@pytest.mark.parametrize(
    "limit",
    [0, 1001],
)
def test_invalid_history_limit(client, limit):

    response = client.get(
        "/predictions/history",
        params={"limit": limit},
        headers=HEADERS,
    )

    assert response.status_code == 422


def test_database_unavailable(
    client,
    monkeypatch,
):

    monkeypatch.setattr(
        api,
        "DATABASE_READY",
        False,
    )

    response = client.get(
        "/predictions/history",
        headers=HEADERS,
    )

    assert response.status_code == 503


def test_prediction_survives_storage_failure(
    client,
    monkeypatch,
):

    monkeypatch.setattr(
        api,
        "save_prediction",
        lambda **kwargs: False,
    )

    features = {
        name: 0.0
        for name in api.FEATURE_NAMES
    }

    response = client.post(
        "/predict",
        headers=HEADERS,
        json={"features": features},
    )

    assert response.status_code == 200

    assert (
        response.json()["stored_in_database"]
        is False
    )
