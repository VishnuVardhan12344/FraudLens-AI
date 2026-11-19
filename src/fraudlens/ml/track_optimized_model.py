
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn


ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "models"

MODEL_PATH = (
    MODEL_DIR / "random_forest_optimized.joblib"
)

METRICS_PATH = (
    MODEL_DIR / "rf_threshold_metrics.json"
)


def track_optimized_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    if not METRICS_PATH.exists():
        raise FileNotFoundError(METRICS_PATH)

    # Load artifacts produced by Lesson 9.
    artifact = joblib.load(MODEL_PATH)

    with open(METRICS_PATH, "r") as file:
        report = json.load(file)

    model = artifact["model"]
    threshold = float(artifact["threshold"])

    metrics = report["test_metrics"]

    # Store experiment metadata locally.
    mlflow.set_tracking_uri(
        f"sqlite:///{ROOT / 'mlflow.db'}"
    )

    mlflow.set_experiment(
        "FraudLens-Fraud-Detection"
    )

    with mlflow.start_run(
        run_name="RandomForest-Optimized"
    ):

        mlflow.log_param(
            "model_type",
            "RandomForestClassifier"
        )

        mlflow.log_param(
            "threshold",
            threshold
        )

        mlflow.log_param(
            "evaluation_protocol",
            "stratified_60_20_20"
        )

        mlflow.log_params({
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "class_weight": model.class_weight,
            "random_state": model.random_state,
        })

        # Log the metrics from the existing test run.
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(
                    name,
                    float(value)
                )

        # Preserve original artifacts.
        mlflow.log_artifact(
            str(METRICS_PATH)
        )

        mlflow.log_artifact(
            str(MODEL_PATH)
        )

        print("\nMLflow tracking completed!")
        print("Threshold:", threshold)
        print("Precision:", metrics["precision"])
        print("Recall:", metrics["recall"])
        print("F1:", metrics["f1"])


if __name__ == "__main__":
    track_optimized_model()
