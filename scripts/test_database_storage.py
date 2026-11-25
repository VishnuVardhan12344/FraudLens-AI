
from uuid import uuid4

from fraudlens.database import initialize_database

from fraudlens.prediction_repository import (
    save_prediction,
    get_prediction_history,
    get_database_summary,
)


def main():

    assert initialize_database(), (
        "Database is not configured"
    )

    request_id = str(uuid4())

    success = save_prediction(
        request_id=request_id,
        probability=0.87,
        threshold=0.35,
        prediction=1,
        label="Fraud",
        model_version="RandomForest-Optimized",
        latency_ms=25.4,
    )

    assert success, "Prediction storage failed"

    print("PASS: Prediction saved")

    history = get_prediction_history(limit=10)

    assert any(
        record["request_id"] == request_id
        for record in history
    ), "Saved prediction not found"

    print("PASS: Prediction retrieved")

    print("\nPrediction history:")
    for record in history:
        print(record)

    print("\nDatabase summary:")
    print(get_database_summary())

    print("\nALL DATABASE TESTS PASSED")


if __name__ == "__main__":
    main()
