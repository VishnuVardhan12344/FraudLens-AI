
import hashlib
import json
from pathlib import Path

import joblib
import mlflow


ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "models"

DATA_PATH = (
    ROOT / "data" / "processed"
    / "creditcard_features.parquet"
)

EXPERIMENTS = [
    {
        "name": "LogisticRegression-Optimized",
        "model_file": "fraud_model_optimized.joblib",
        "metrics_file": "threshold_metrics.json",
    },
    {
        "name": "RandomForest-Optimized",
        "model_file": "random_forest_optimized.joblib",
        "metrics_file": "rf_threshold_metrics.json",
    },
]


def file_hash(path):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def track_experiments():

    mlflow.set_tracking_uri(
        f"sqlite:///{ROOT / 'mlflow.db'}"
    )

    mlflow.set_experiment(
        "FraudLens-Fraud-Detection"
    )

    for experiment in EXPERIMENTS:

        model_path = (
            MODEL_DIR / experiment["model_file"]
        )

        metrics_path = (
            MODEL_DIR / experiment["metrics_file"]
        )

        if not model_path.exists():
            print("Skipping missing model:", model_path)
            continue

        if not metrics_path.exists():
            print("Skipping missing metrics:", metrics_path)
            continue

        artifact = joblib.load(model_path)

        with open(metrics_path, "r") as f:
            report = json.load(f)

        model = artifact["model"]
        threshold = float(artifact["threshold"])

        metrics = report["test_metrics"]

        with mlflow.start_run(
            run_name=experiment["name"]
        ):

            mlflow.log_param(
                "model_type",
                type(model).__name__,
            )

            mlflow.log_param(
                "threshold",
                threshold,
            )

            mlflow.log_param(
                "evaluation_protocol",
                "stratified_60_20_20",
            )

            mlflow.set_tag(
                "project",
                "FraudLens AI",
            )

            mlflow.set_tag(
                "evaluation_status",
                "exploratory_not_production_validated",
            )

            if DATA_PATH.exists():
                mlflow.set_tag(
                    "current_dataset_sha256",
                    file_hash(DATA_PATH),
                )

            for key, value in metrics.items():
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                ):
                    mlflow.log_metric(
                        key,
                        float(value),
                    )

            mlflow.log_artifact(str(model_path))
            mlflow.log_artifact(str(metrics_path))

            print("\nTracked:", experiment["name"])
            print("Threshold:", threshold)
            print("Precision:", metrics["precision"])
            print("Recall:", metrics["recall"])
            print("F1:", metrics["f1"])

    print("\nExperiment tracking complete!")


if __name__ == "__main__":
    track_experiments()
