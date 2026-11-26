
from types import SimpleNamespace
from unittest.mock import MagicMock

import joblib
import pytest

from fraudlens import model_registry as registry


class FakeModel:

    classes_ = [0, 1]

    def predict_proba(self, X):
        return [[0.20, 0.80] for _ in X]


@pytest.fixture
def valid_bundle():

    return {
        "model": FakeModel(),
        "threshold": 0.35,
        "feature_names": (
            registry.FEATURE_NAMES
            + registry.ENGINEERED_FEATURES
        ),
    }


@pytest.fixture
def mock_client(monkeypatch):

    client = MagicMock()

    client.get_registered_model.return_value = (
        SimpleNamespace(
            aliases={
                "champion": "1",
                "candidate": "2",
            }
        )
    )

    monkeypatch.setattr(
        registry,
        "get_client",
        lambda: client,
    )

    return client


def test_valid_bundle(valid_bundle):

    assert registry.validate_bundle(valid_bundle)


def test_missing_model(valid_bundle):

    del valid_bundle["model"]

    with pytest.raises(ValueError):
        registry.validate_bundle(valid_bundle)


def test_invalid_threshold(valid_bundle):

    valid_bundle["threshold"] = 1.5

    with pytest.raises(ValueError):
        registry.validate_bundle(valid_bundle)


def test_nan_threshold(valid_bundle):

    valid_bundle["threshold"] = float("nan")

    with pytest.raises(ValueError):
        registry.validate_bundle(valid_bundle)


def test_invalid_features(valid_bundle):

    valid_bundle["feature_names"] = ["Amount"]

    with pytest.raises(ValueError):
        registry.validate_bundle(valid_bundle)


def test_duplicate_features(valid_bundle):

    valid_bundle["feature_names"].append("Amount")

    with pytest.raises(ValueError):
        registry.validate_bundle(valid_bundle)


def test_screening_passed():

    metrics = {
        "precision": 0.85,
        "recall": 0.82,
        "f1": 0.83,
    }

    assert registry.passes_screening(metrics)


def test_screening_failed():

    metrics = {
        "precision": 0.40,
        "recall": 0.90,
        "f1": 0.55,
    }

    assert not registry.passes_screening(metrics)


def test_missing_metrics():

    assert not registry.passes_screening({})


def test_invalid_metric():

    metrics = {
        "precision": float("nan"),
        "recall": 0.90,
        "f1": 0.80,
    }

    assert not registry.passes_screening(metrics)


def test_alias_exists(mock_client):

    version = registry.get_alias_version(
        mock_client,
        "champion",
    )

    assert version == "1"


def test_alias_missing(mock_client):

    version = registry.get_alias_version(
        mock_client,
        "previous",
    )

    assert version is None


def test_promotion_requires_approval():

    with pytest.raises(PermissionError):

        registry.promote_candidate(
            "2",
            approved=False,
        )


def test_candidate_promotion(mock_client):

    mock_client.get_model_version.return_value = (
        SimpleNamespace(
            status="READY",
            tags={
                "screening_passed": "true",
                "model_role": "candidate",
            },
        )
    )

    result = registry.promote_candidate(
        "2",
        approved=True,
    )

    assert result["champion"] == "2"
    assert result["previous"] == "1"
    assert result["action"] == "promoted"

    mock_client.set_registered_model_alias.assert_any_call(
        registry.MODEL_NAME,
        "previous",
        "1",
    )

    mock_client.set_registered_model_alias.assert_any_call(
        registry.MODEL_NAME,
        "champion",
        "2",
    )


def test_rejected_candidate(mock_client):

    mock_client.get_model_version.return_value = (
        SimpleNamespace(
            tags={
                "screening_passed": "false",
                "model_role": "candidate",
            }
        )
    )

    with pytest.raises(ValueError):

        registry.promote_candidate(
            "2",
            approved=True,
        )

    mock_client.set_registered_model_alias.assert_not_called()


def test_rollback(mock_client):

    mock_client.get_registered_model.return_value = (
        SimpleNamespace(
            aliases={
                "champion": "2",
                "previous": "1",
            }
        )
    )

    result = registry.rollback_model()

    assert result["champion"] == "1"
    assert result["previous"] == "2"
    assert result["action"] == "rolled_back"


def test_rollback_without_previous(mock_client):

    with pytest.raises(ValueError):
        registry.rollback_model()


def test_deploy_champion(
    mock_client,
    valid_bundle,
    monkeypatch,
    tmp_path,
):

    artifact = tmp_path / "downloaded.joblib"
    destination = tmp_path / "active.joblib"
    backup_dir = tmp_path / "backups"

    joblib.dump(valid_bundle, artifact)
    joblib.dump(valid_bundle, destination)

    mock_client.get_model_version.return_value = (
        SimpleNamespace(
            tags={"bundle_run_id": "test-run"}
        )
    )

    monkeypatch.setattr(
        registry,
        "BACKUP_DIR",
        backup_dir,
    )

    monkeypatch.setattr(
        registry.mlflow.artifacts,
        "download_artifacts",
        lambda **kwargs: str(artifact),
    )

    result = registry.deploy_champion(
        destination=destination
    )

    deployed = joblib.load(destination)

    assert result["deployed_version"] == "1"
    assert deployed["registry_version"] == "1"
    assert deployed["deployed_at"]
    assert deployed["threshold"] == 0.35

    assert len(
        list(backup_dir.glob("*.joblib"))
    ) == 1


def test_deploy_invalid_bundle(
    mock_client,
    monkeypatch,
    tmp_path,
):

    artifact = tmp_path / "invalid.joblib"
    destination = tmp_path / "active.joblib"

    joblib.dump({}, artifact)

    destination.write_bytes(b"original-model")

    mock_client.get_model_version.return_value = (
        SimpleNamespace(
            tags={"bundle_run_id": "test-run"}
        )
    )

    monkeypatch.setattr(
        registry,
        "BACKUP_DIR",
        tmp_path / "backups",
    )

    monkeypatch.setattr(
        registry.mlflow.artifacts,
        "download_artifacts",
        lambda **kwargs: str(artifact),
    )

    with pytest.raises(ValueError):

        registry.deploy_champion(
            destination=destination
        )

    assert destination.read_bytes() == b"original-model"

