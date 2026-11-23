
import os
import sys
import time

import httpx

# Docker API address
BASE_URL = os.getenv(
    "FRAUDLENS_API_URL",
    "http://127.0.0.1:8001"
)


def test_health():
    print("Checking FraudLens API health...")

    # Wait up to approximately 60 seconds
    for attempt in range(30):
        try:
            response = httpx.get(
                f"{BASE_URL}/health",
                timeout=3
            )

            if response.status_code == 200:
                data = response.json()

                if data.get("status") == "ready":
                    print("PASS: /health")
                    return

        except (httpx.RequestError, ValueError):
            pass

        time.sleep(2)

    raise RuntimeError(
        "FraudLens API did not become ready"
    )


def test_prediction():
    print("Testing fraud prediction...")

    # Synthetic transaction with 30 input fields
    transaction = {
        "Time": 1000.0,
        "Amount": 120.0
    }

    for i in range(1, 29):
        transaction[f"V{i}"] = 0.0

    response = httpx.post(
    f"{BASE_URL}/predict",
    json={"features": transaction},
    timeout=15
    )

    print("Status code:", response.status_code)
    print("Response:", response.text)

    response.raise_for_status()

    result = response.json()

    assert isinstance(result, dict), (
        "Prediction must return a JSON object"
    )

    print("PASS: /predict")


if __name__ == "__main__":
    try:
        test_health()
        test_prediction()

        print("All Docker smoke tests passed!")

    except Exception as error:
        print(f"Smoke test failed: {error}")
        sys.exit(1)
