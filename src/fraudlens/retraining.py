
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)

from fraudlens.drift import FEATURE_NAMES


ROOT = Path(__file__).resolve().parents[2]

CANDIDATE_DIR = ROOT / "models" / "candidates"

ENGINEERED_FEATURES = [
    "log_amount",
    "amount_is_zero",
    "elapsed_day",
    "elapsed_hour",
]


def prepare_training_features(df):

    X = df[FEATURE_NAMES].copy()

    if not np.isfinite(
        X.to_numpy(dtype=float)
    ).all():
        raise ValueError("Invalid feature values")

    if (
        (X["Amount"] < 0).any()
        or (X["Time"] < 0).any()
    ):
        raise ValueError("Negative amount or time")

    X["log_amount"] = np.log1p(
        X["Amount"]
    )

    X["amount_is_zero"] = (
        X["Amount"] == 0
    ).astype(int)

    X["elapsed_day"] = (
        X["Time"] // 86400
    ).astype(int)

    X["elapsed_hour"] = (
        (X["Time"] % 86400) // 3600
    ).astype(int)

    return X


def evaluate_model(
    model,
    X,
    y,
    threshold,
):

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "precision": float(
            precision_score(
                y, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y, predictions, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y, predictions, zero_division=0
            )
        ),
        "average_precision": float(
            average_precision_score(
                y, probabilities
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y, probabilities
            )
        ),
    }


def select_threshold(model, X, y):

    probabilities = model.predict_proba(X)[:, 1]

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in np.arange(
        0.05, 1.0, 0.05
    ):

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y,
            predictions,
            zero_division=0,
        )

        if score > best_f1:

            best_f1 = score

            best_threshold = float(
                round(threshold, 2)
            )

    return best_threshold


def train_candidate(
    data_path,
    output_dir=CANDIDATE_DIR,
):

    df = pd.read_csv(data_path)

    if "Class" not in df.columns:
        raise ValueError(
            "Retraining requires actual fraud labels"
        )

    if not df["Class"].isin([0, 1]).all():
        raise ValueError("Invalid target labels")

    df = df.drop_duplicates().reset_index(drop=True)

    if df["Class"].value_counts().min() < 50:
        raise ValueError(
            "Insufficient labeled class examples"
        )

    if df["Class"].nunique() != 2:
        raise ValueError("Both classes are required")

    X = prepare_training_features(df)

    y = df["Class"].astype(int)

    # 60% training, 20% validation, 20% testing

    X_train, X_remaining, y_train, y_remaining = (
        train_test_split(
            X,
            y,
            test_size=0.40,
            stratify=y,
            random_state=42,
        )
    )

    X_val, X_test, y_val, y_test = (
        train_test_split(
            X_remaining,
            y_remaining,
            test_size=0.50,
            stratify=y_remaining,
            random_state=42,
        )
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    print("Training candidate Random Forest...")

    model.fit(X_train, y_train)

    threshold = select_threshold(
        model,
        X_val,
        y_val,
    )

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
        threshold,
    )

    # Illustrative candidate screening gates

    passed_screening = (
        metrics["precision"] >= 0.70
        and metrics["recall"] >= 0.70
        and metrics["f1"] >= 0.75
    )

    version = (
        datetime.now(timezone.utc)
        .strftime("%Y%m%d_%H%M%S")
        + "_"
        + uuid4().hex[:8]
    )

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        output_dir
        / f"random_forest_{version}.joblib"
    )

    bundle = {
        "model": model,
        "threshold": threshold,
        "feature_names": list(X.columns),
        "version": version,
        "metrics": metrics,
        "passed_screening": passed_screening,
        "status": "candidate",
    }

    joblib.dump(bundle, model_path)

    return {
        "version": version,
        "model_path": str(model_path),
        "threshold": threshold,
        "metrics": metrics,
        "passed_screening": passed_screening,
        "training_records": len(X_train),
        "validation_records": len(X_val),
        "test_records": len(X_test),
        "status": "candidate",
    }
