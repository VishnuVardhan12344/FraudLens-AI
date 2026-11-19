
import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "creditcard_features.parquet"
)

MODEL_DIR = PROJECT_ROOT / "models"


def load_ml_data():
    """Load processed transaction data."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_PATH}"
        )

    df = pd.read_parquet(DATA_PATH)

    logger.info("Loaded %d transactions", len(df))

    if "Class" not in df.columns:
        raise ValueError("Missing target column: Class")

    if df["Class"].isna().any():
        raise ValueError("Target contains missing values")

    if not df["Class"].isin([0, 1]).all():
        raise ValueError("Target must contain only 0 and 1")

    # Keep the original features and engineered features.
    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    if not all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes):
        raise ValueError("All model features must be numeric")

    if X.isna().any().any():
        raise ValueError("Model features contain missing values")

    if y.value_counts().min() < 2 or y.nunique() != 2:
        raise ValueError(
            "Both classes require at least two records"
        )

    return X, y


def evaluate_model(model, X_test, y_test):
    """Evaluate a trained fraud detection model."""

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "precision": float(
            precision_score(
                y_test, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_test, predictions, zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y_test, predictions, zero_division=0
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_test, probabilities
            )
        ),
    }

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    return metrics


def train_models():
    """Train and compare two fraud detection models."""

    X, y = load_ml_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    logger.info(
        "Training rows: %d | Testing rows: %d",
        len(X_train),
        len(X_test),
    )

    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = {}

    for name, model in models.items():

        logger.info("Training model: %s", name)

        model.fit(X_train, y_train)

        metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        results[name] = metrics

        print(f"\n{name} results:")
        print(json.dumps(metrics, indent=2))

    # Learning-stage comparison only:
    # A future lesson will use a validation set for
    # model selection and preserve the test set.
    best_name = max(
        results,
        key=lambda name: results[name]["pr_auc"],
    )

    best_model = models[best_name]

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = MODEL_DIR / "fraud_model.joblib"

    joblib.dump(best_model, model_path)

    logger.info(
        "Saved model %s to %s",
        best_name,
        model_path,
    )

    metrics_path = MODEL_DIR / "model_metrics.json"

    metrics_path.write_text(
        json.dumps(
            {
                "selected_model": best_name,
                "metrics": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nSelected model:", best_name)
    print("Saved model:", model_path)


if __name__ == "__main__":
    train_models()
