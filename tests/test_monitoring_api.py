
import numpy as np
import pytest

from fastapi.testclient import TestClient

from fraudlens import api
from fraudlens import rate_limiter as limiter_module

from fraudlens.rate_limiter import RateLimiter
from fraudlens.prediction_monitor import PredictionMonitor


TEST_KEY = "fraudlens-monitoring-test-key"

HEADERS = {
    "X-API-Key": TEST_KEY
}


class FakeModel:

    def predict_proba(self, X):

        return np.array([
            [0.90, 0.10]
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
            max_requests=2,
            window_seconds=60,
        ),
    )

    bundle = {
        "model": FakeModel(),
        "threshold": 0.35,
        "feature_names": (
            api.FEATURE_NAMES
            + api.ENGINEERED_FEATURES
        ),
    }

    monkeypatch.setattr(
        api,
        "load_model",
        lambda: bundle,
    )

    with TestClient(api.app) as test_client:

        yield test_client


@pytest.fixture
def payload():

    features = {
        name: 0.0
        for name in api.FEATURE_NAMES
    }

    features["Time"] = 1000.0
    features["Amount"] = 120.0

    return {
        "features": features
    }


def test_request_monitoring(client):

    response = client.get("/health")

    assert response.status_code == 200

    assert "X-Request-ID" in response.headers

    assert float(
        response.headers["X-Response-Time-MS"]
    ) >= 0


def test_monitoring_requires_authentication(client):

    response = client.get(
        "/monitoring/predictions"
    )

    assert response.status_code == 401


def test_prediction_counter(client, payload):

    response = client.post(
        "/predict",
        json=payload,
        headers=HEADERS,
    )

    assert response.status_code == 200

    response = client.get(
        "/monitoring/predictions",
        headers=HEADERS,
    )

    metrics = response.json()

    assert metrics["total_predictions"] == 1

    assert metrics["legitimate_predictions"] == 1


def test_rate_limit(client, payload):

    for _ in range(2):

        response = client.post(
            "/predict",
            json=payload,
            headers=HEADERS,
        )

        assert response.status_code == 200

    response = client.post(
        "/predict",
        json=payload,
        headers=HEADERS,
    )

    assert response.status_code == 429

    assert "Retry-After" in response.headers

    assert client.get("/health").status_code == 200


def test_prometheus_requires_authentication(client):

    response = client.get("/metrics")

    assert response.status_code == 401


def test_prometheus_metrics(client):

    response = client.get(
        "/metrics",
        headers=HEADERS,
    )

    assert response.status_code == 200

    assert (
        "fraudlens_model_predictions_total"
        in response.text
    )

    assert (
        "fraudlens_api_requests_total"
        in response.text
    )
