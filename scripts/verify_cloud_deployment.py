
import os
import time

import httpx

from dotenv import load_dotenv

load_dotenv()

BASE_URL = (
    "https://fraudlens-ai-ecn5.onrender.com"
)

API_KEY = os.getenv("FRAUDLENS_API_KEY")

if not API_KEY:
    raise RuntimeError("API key not configured")

HEADERS = {
    "X-API-Key": API_KEY
}


def verify_deployment():

    start = time.perf_counter()

    with httpx.Client(
        timeout=120,
        trust_env=False,
    ) as client:

        response = client.get(
            f"{BASE_URL}/health"
        )

        response.raise_for_status()

        assert response.json()["status"] == "ready"

        print("PASS: Cloud API health")

        transaction = {
            "Time": 3600.0,
            "Amount": 120.0,
        }

        transaction.update({
            f"V{i}": 0.0
            for i in range(1, 29)
        })

        payload = {
            "features": transaction
        }

        response = client.post(
            f"{BASE_URL}/predict",
            json=payload,
        )

        assert response.status_code == 401

        print("PASS: Authentication required")

        response = client.post(
            f"{BASE_URL}/predict",
            headers=HEADERS,
            json=payload,
        )

        response.raise_for_status()

        prediction = response.json()

        assert 0 <= prediction[
            "fraud_probability"
        ] <= 1

        assert prediction["prediction"] in [0, 1]

        expected_label = (
            "Fraud"
            if prediction["prediction"] == 1
            else "Legitimate"
        )

        assert prediction["label"] == expected_label

        print("PASS: Cloud prediction")
        print(prediction)

        response = client.get(
            f"{BASE_URL}/model/info",
            headers=HEADERS,
        )

        response.raise_for_status()

        info = response.json()

        assert info["feature_count"] == 34

        print("PASS: Model information")
        print(info)

        response = client.get(
            f"{BASE_URL}/monitoring/predictions",
            headers=HEADERS,
        )

        response.raise_for_status()

        print("PASS: Prediction monitoring")

        response = client.get(
            f"{BASE_URL}/metrics",
            headers=HEADERS,
        )

        response.raise_for_status()

        assert (
            "fraudlens_api_requests_total"
            in response.text
        )

        print("PASS: Prometheus metrics")

    elapsed = time.perf_counter() - start

    print(
        f"\nVerification completed in "
        f"{elapsed:.2f} seconds"
    )

    print("SUCCESS: Cloud verification passed")


if __name__ == "__main__":
    verify_deployment()
