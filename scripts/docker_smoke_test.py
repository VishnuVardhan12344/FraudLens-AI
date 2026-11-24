
import os
import time

import httpx
from dotenv import load_dotenv


load_dotenv()

BASE_URL = os.getenv(
    "FRAUDLENS_API_URL",
    "http://127.0.0.1:8001",
).rstrip("/")

API_KEY = os.getenv("FRAUDLENS_API_KEY")

if not API_KEY:
    raise RuntimeError("API key is not configured")


# 1. Wait for the API to become ready.

def wait_for_api():

    for attempt in range(30):

        try:
            response = httpx.get(
                f"{BASE_URL}/health",
                timeout=3,
            )

            if (
                response.status_code == 200
                and response.json().get("status") == "ready"
            ):
                print("Health check passed")
                return

        except httpx.RequestError:
            pass

        print(f"Waiting for API: attempt {attempt + 1}")
        time.sleep(2)

    raise RuntimeError("API did not become ready")


# 2. Create a synthetic transaction.

def build_payload():

    transaction = {
        "Time": 1000.0,
        "Amount": 120.0,
    }

    transaction.update({
        f"V{i}": 0.0
        for i in range(1, 29)
    })

    return {"features": transaction}


# 3. Test API authentication.

def test_authentication():

    payload = build_payload()

    # Missing API key
    response = httpx.post(
        f"{BASE_URL}/predict",
        json=payload,
        timeout=30,
    )

    assert response.status_code == 401
    print("Missing API key test passed")

    # Incorrect API key
    response = httpx.post(
        f"{BASE_URL}/predict",
        json=payload,
        headers={
            "X-API-Key": "incorrect-key"
        },
        timeout=30,
    )

    assert response.status_code == 401
    print("Invalid API key test passed")

    # Correct API key
    response = httpx.post(
        f"{BASE_URL}/predict",
        json=payload,
        headers={
            "X-API-Key": API_KEY
        },
        timeout=30,
    )

    response.raise_for_status()

    result = response.json()

    assert result["prediction"] in (0, 1)
    assert 0 <= result["fraud_probability"] <= 1

    print("Authenticated prediction passed")
    print(result)


# 4. Execute all smoke tests.

if __name__ == "__main__":

    wait_for_api()
    test_authentication()

    print("All Docker smoke tests passed")
