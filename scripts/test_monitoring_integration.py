
import os

import httpx

from dotenv import load_dotenv


load_dotenv()

BASE_URL = "http://127.0.0.1:8000"

API_KEY = os.getenv("FRAUDLENS_API_KEY")

if not API_KEY:
    raise RuntimeError("Missing API key")

headers = {
    "X-API-Key": API_KEY
}

features = {
    "Time": 1000.0,
    "Amount": 120.0,
}

for i in range(1, 29):
    features[f"V{i}"] = 0.0

payload = {
    "features": features
}


with httpx.Client(timeout=30) as client:

    # Test unauthorized request
    response = client.post(
        f"{BASE_URL}/predict",
        json=payload,
    )

    assert response.status_code == 401

    print("PASS: Authentication")

    # Test three allowed requests
    for i in range(3):

        response = client.post(
            f"{BASE_URL}/predict",
            json=payload,
            headers=headers,
        )

        print(
            f"Request {i + 1}:",
            response.status_code,
        )

        assert response.status_code == 200

    # Fourth request must be rejected
    response = client.post(
        f"{BASE_URL}/predict",
        json=payload,
        headers=headers,
    )

    assert response.status_code == 429
    assert "Retry-After" in response.headers

    print("PASS: Rate limiting")

    # Health endpoint must remain available
    response = client.get(
        f"{BASE_URL}/health"
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ready"

    print("PASS: Health endpoint")

    # Verify metrics authentication
    response = client.get(
        f"{BASE_URL}/metrics"
    )

    assert response.status_code == 401

    print("PASS: Metrics authentication")

    # Retrieve Prometheus metrics
    response = client.get(
        f"{BASE_URL}/metrics",
        headers=headers,
    )

    assert response.status_code == 200

    assert (
        "fraudlens_api_requests_total"
        in response.text
    )

    assert (
        "fraudlens_model_predictions_total"
        in response.text
    )

    assert (
        "fraudlens_rate_limit_rejections_total 1.0"
        in response.text
    )

    print("PASS: Prometheus metrics")

    # Retrieve prediction monitoring statistics
    response = client.get(
        f"{BASE_URL}/monitoring/predictions",
        headers=headers,
    )

    assert response.status_code == 200

    metrics = response.json()

    assert metrics["total_predictions"] == 3

    print("PASS: Prediction monitoring")
    print(metrics)


print("\nALL MONITORING INTEGRATION TESTS PASSED")

