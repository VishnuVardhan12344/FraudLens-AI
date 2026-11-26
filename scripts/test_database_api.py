
import os

import httpx

from dotenv import load_dotenv


load_dotenv()

BASE_URL = "http://127.0.0.1:8000"

API_KEY = os.getenv("FRAUDLENS_API_KEY")

assert API_KEY, "Missing API key"

HEADERS = {
    "X-API-Key": API_KEY
}


def main():

    with httpx.Client(
        base_url=BASE_URL,
        timeout=30,
    ) as client:

        # Test database connectivity

        response = client.get(
            "/database/status",
            headers=HEADERS,
        )

        assert response.status_code == 200
        assert response.json()["status"] == "ready"

        print("PASS: Database ready")

        # Test unauthorized access

        response = client.get(
            "/predictions/history"
        )

        assert response.status_code == 401

        print("PASS: History authentication")

        # Test prediction storage

        features = {
            "Time": 1000.0,
            "Amount": 120.0,
        }

        for i in range(1, 29):
            features[f"V{i}"] = 0.0

        response = client.post(
            "/predict",
            json={"features": features},
            headers=HEADERS,
        )

        assert response.status_code == 200, (
            response.text
        )

        prediction = response.json()

        assert prediction["stored_in_database"]

        request_id = prediction["request_id"]

        print("PASS: Prediction saved")

        # Test history endpoint

        response = client.get(
            "/predictions/history",
            params={"limit": 100},
            headers=HEADERS,
        )

        assert response.status_code == 200

        records = response.json()["predictions"]

        assert any(
            record["request_id"] == request_id
            for record in records
        )

        print("PASS: Prediction history")

        # Test summary endpoint

        response = client.get(
            "/predictions/summary",
            headers=HEADERS,
        )

        assert response.status_code == 200

        summary = response.json()

        assert summary["total_predictions"] >= 1

        print("PASS: Database summary")

        # Test limit validation

        response = client.get(
            "/predictions/history",
            params={"limit": 1001},
            headers=HEADERS,
        )

        assert response.status_code == 422

        print("PASS: History limit validation")

        print("\nDatabase summary:")
        print(summary)

        print("\nALL DATABASE API TESTS PASSED")


if __name__ == "__main__":
    main()
