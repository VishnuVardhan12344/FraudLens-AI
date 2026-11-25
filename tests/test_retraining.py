
import joblib
import numpy as np
import pandas as pd
import pytest

from fraudlens.drift import FEATURE_NAMES

from fraudlens.retraining import (
    prepare_training_features,
    evaluate_model,
    select_threshold,
    train_candidate,
)


class FakeModel:

    def predict_proba(self, X):

        probability = np.array([
            0.10, 0.20, 0.80, 0.90
        ])

        return np.column_stack([
            1 - probability,
            probability,
        ])


def test_feature_engineering():

    df = pd.DataFrame({
        name: [0.0, 1.0]
        for name in FEATURE_NAMES
    })

    df["Time"] = [0, 90000]
    df["Amount"] = [0, 100]

    result = prepare_training_features(df)

    assert result.shape == (2, 34)

    assert result.iloc[0]["amount_is_zero"] == 1

    assert result.iloc[1]["elapsed_day"] == 1

    assert result.iloc[1]["elapsed_hour"] == 1


def test_model_evaluation():

    X = pd.DataFrame({
        "feature": [1, 2, 3, 4]
    })

    y = np.array([0, 0, 1, 1])

    metrics = evaluate_model(
        FakeModel(),
        X,
        y,
        threshold=0.50,
    )

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_threshold_selection():

    X = pd.DataFrame({
        "feature": [1, 2, 3, 4]
    })

    y = np.array([0, 0, 1, 1])

    threshold = select_threshold(
        FakeModel(),
        X,
        y,
    )

    assert 0.20 < threshold <= 0.80


def test_missing_labels(tmp_path):

    path = tmp_path / "unlabeled.csv"

    pd.DataFrame({
        "Amount": [100, 200]
    }).to_csv(path, index=False)

    with pytest.raises(ValueError):

        train_candidate(
            path,
            output_dir=tmp_path,
        )


def test_candidate_training(tmp_path):

    rng = np.random.default_rng(42)

    size = 400

    df = pd.DataFrame(
        rng.normal(
            size=(size, len(FEATURE_NAMES))
        ),
        columns=FEATURE_NAMES,
    )

    df["Time"] = np.arange(size) * 100

    df["Amount"] = np.abs(
        df["Amount"]
    ) * 100

    df["Class"] = (
        df["V1"] > 0
    ).astype(int)

    path = tmp_path / "labeled.csv"

    df.to_csv(path, index=False)

    result = train_candidate(
        path,
        output_dir=tmp_path / "candidates",
    )

    bundle = joblib.load(
        result["model_path"]
    )

    assert result["status"] == "candidate"

    assert len(bundle["feature_names"]) == 34

    assert 0 <= bundle["threshold"] <= 1

    assert "metrics" in bundle
