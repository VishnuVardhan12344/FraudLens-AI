
import os
import math
import tempfile

from pathlib import Path
from datetime import datetime, timezone

import joblib
import mlflow

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from fraudlens.drift import FEATURE_NAMES
from fraudlens.retraining import ENGINEERED_FEATURES


ROOT = Path(__file__).resolve().parents[2]

MODEL_NAME = "FraudLens-RandomForest"

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5050",
)

EXPERIMENT_NAME = "FraudLens-Model-Registry"

ACTIVE_MODEL_PATH = (
    ROOT / "models" / "random_forest_optimized.joblib"
)

BACKUP_DIR = ROOT / "models" / "backups"


def get_client():

    mlflow.set_tracking_uri(TRACKING_URI)

    return MlflowClient(
        tracking_uri=TRACKING_URI
    )


def validate_bundle(bundle):

    required = {
        "model",
        "threshold",
        "feature_names",
    }

    if not isinstance(bundle, dict):
        raise ValueError("Invalid model bundle")

    if not required.issubset(bundle):
        raise ValueError("Missing model bundle keys")

    threshold = float(bundle["threshold"])

    if not math.isfinite(threshold):
        raise ValueError("Invalid threshold")

    if not 0 <= threshold <= 1:
        raise ValueError("Invalid threshold")

    features = list(bundle["feature_names"])

    expected = set(
        FEATURE_NAMES + ENGINEERED_FEATURES
    )

    if (
        len(features) != len(set(features))
        or set(features) != expected
    ):
        raise ValueError("Model features do not match")

    model = bundle["model"]

    if list(model.classes_) != [0, 1]:
        raise ValueError("Invalid model classes")

    if not hasattr(model, "predict_proba"):
        raise ValueError("Missing predict_proba")

    return True


def passes_screening(metrics):

    required = {
        "precision": 0.70,
        "recall": 0.70,
        "f1": 0.75,
    }

    for name, minimum in required.items():

        if name not in metrics:
            return False

        value = float(metrics[name])

        if not math.isfinite(value):
            return False

        if not minimum <= value <= 1:
            return False

    return True



def get_alias_version(client, alias):
    registered_model = client.get_registered_model(
        MODEL_NAME
    )

    version = registered_model.aliases.get(alias)

    if version is None:
        return None

    return str(version)


def register_model_bundle(
    model_path,
    baseline=False,
):

    model_path = Path(model_path).resolve()

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    bundle = joblib.load(model_path)

    validate_bundle(bundle)

    client = get_client()

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    metrics = bundle.get("metrics", {})

    screened = passes_screening(metrics)

    with mlflow.start_run(
        run_name=(
            "baseline-model"
            if baseline
            else "candidate-model"
        )
    ) as run:

        mlflow.log_param(
            "threshold",
            float(bundle["threshold"]),
        )

        mlflow.log_param(
            "feature_count",
            len(bundle["feature_names"]),
        )

        mlflow.log_param(
            "model_type",
            type(bundle["model"]).__name__,
        )

        for name, value in metrics.items():

            if isinstance(value, (int, float)):

                if math.isfinite(value):

                    mlflow.log_metric(
                        name,
                        float(value),
                    )

        mlflow.set_tag(
            "screening_passed",
            str(screened).lower(),
        )

        model_info = mlflow.sklearn.log_model(
    sk_model=bundle["model"],
    name="model",
    serialization_format="skops",
    skops_trusted_types=[
        "sklearn.tree._tree.Tree"
    ],
)

        with tempfile.TemporaryDirectory() as temp:

            bundle_path = (
                Path(temp) / "model_bundle.joblib"
            )

            joblib.dump(bundle, bundle_path)

            mlflow.log_artifact(
                str(bundle_path),
                artifact_path="fraudlens_bundle",
            )

        registered = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=MODEL_NAME,
        )

        version = str(registered.version)

        client.set_model_version_tag(
            MODEL_NAME,
            version,
            "screening_passed",
            str(screened).lower(),
        )

        client.set_model_version_tag(
            MODEL_NAME,
            version,
            "model_role",
            "baseline" if baseline else "candidate",
        )

        client.set_model_version_tag(
            MODEL_NAME,
            version,
            "bundle_run_id",
            run.info.run_id,
        )

        if baseline:

            if get_alias_version(
                client, "champion"
            ) is None:

                client.set_registered_model_alias(
                    MODEL_NAME,
                    "champion",
                    version,
                )

        else:

            client.set_registered_model_alias(
                MODEL_NAME,
                "candidate",
                version,
            )

    return {
        "model_name": MODEL_NAME,
        "version": version,
        "run_id": run.info.run_id,
        "screening_passed": screened,
        "baseline": baseline,
    }


def promote_candidate(
    version,
    approved=False,
):

    if not approved:
        raise PermissionError(
            "Explicit approval is required"
        )

    client = get_client()

    version = str(version)

    model_version = client.get_model_version(
        MODEL_NAME,
        version,
    )

    if model_version.tags.get(
        "screening_passed"
    ) != "true":

        raise ValueError(
            "Candidate failed screening"
        )

    if model_version.tags.get(
        "model_role"
    ) != "candidate":

        raise ValueError(
            "Selected version is not a candidate"
        )

    current = get_alias_version(
        client,
        "champion",
    )

    if current == version:

        return {
            "champion": version,
            "action": "already_promoted",
        }

    if current is not None:

        client.set_registered_model_alias(
            MODEL_NAME,
            "previous",
            current,
        )

    client.set_registered_model_alias(
        MODEL_NAME,
        "champion",
        version,
    )

    client.set_model_version_tag(
        MODEL_NAME,
        version,
        "approved",
        "true",
    )

    return {
        "previous": current,
        "champion": version,
        "action": "promoted",
    }


def rollback_model():

    client = get_client()

    current = get_alias_version(
        client,
        "champion",
    )

    previous = get_alias_version(
        client,
        "previous",
    )

    if previous is None:

        raise ValueError(
            "No previous model available"
        )

    if current is None or current == previous:

        raise ValueError(
            "No distinct champion to roll back"
        )

    client.set_registered_model_alias(
        MODEL_NAME,
        "champion",
        previous,
    )

    client.set_registered_model_alias(
        MODEL_NAME,
        "previous",
        current,
    )

    return {
        "champion": previous,
        "previous": current,
        "action": "rolled_back",
    }


def deploy_champion(
    destination=ACTIVE_MODEL_PATH,
):

    client = get_client()

    version = get_alias_version(
        client,
        "champion",
    )

    if version is None:

        raise ValueError(
            "No champion model registered"
        )

    model_version = client.get_model_version(
        MODEL_NAME,
        version,
    )

    run_id = model_version.tags[
        "bundle_run_id"
    ]

    destination = Path(destination)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    BACKUP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory() as temp:

        artifact = (
            mlflow.artifacts.download_artifacts(
                artifact_uri=(
                    f"runs:/{run_id}/"
                    "fraudlens_bundle/model_bundle.joblib"
                ),
                dst_path=temp,
            )
        )

        bundle = joblib.load(artifact)

        validate_bundle(bundle)

        bundle["registry_version"] = version

        bundle["registered_model_name"] = MODEL_NAME

        bundle["deployed_at"] = (
            datetime.now(timezone.utc).isoformat()
        )

        if destination.exists():

            timestamp = datetime.now(
                timezone.utc
            ).strftime("%Y%m%d_%H%M%S_%f")

            backup_path = (
                BACKUP_DIR
                / f"model_backup_{timestamp}.joblib"
            )

            joblib.dump(
                joblib.load(destination),
                backup_path,
            )

        temporary_path = destination.with_suffix(
            ".tmp"
        )

        try:

            joblib.dump(
                bundle,
                temporary_path,
            )

            os.replace(
                temporary_path,
                destination,
            )

        finally:

            if temporary_path.exists():
                temporary_path.unlink()

    return {
        "deployed_version": version,
        "destination": str(destination),
        "action": "deployed",
    }


def registry_status():

    client = get_client()

    versions = client.search_model_versions(
        f"name = '{MODEL_NAME}'"
    )

    return {
        "model_name": MODEL_NAME,
        "champion": get_alias_version(
            client, "champion"
        ),
        "candidate": get_alias_version(
            client, "candidate"
        ),
        "previous": get_alias_version(
            client, "previous"
        ),
        "versions": [
            {
                "version": str(v.version),
                "status": v.status,
                "run_id": v.run_id,
                "tags": dict(v.tags),
            }
            for v in versions
        ],
    }
