
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "creditcard_features.parquet"
)

MODEL_DIR = PROJECT_ROOT / "models"


def calculate_metrics(y_true, probabilities, threshold):
    """Calculate fraud detection metrics."""

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "precision": float(
            precision_score(
                y_true, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true, predictions, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true, predictions, zero_division=0
            )
        ),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "total_alerts": int(tp + fp),
    }


def optimize_threshold():
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    df = pd.read_parquet(DATA_PATH)

    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    if y.nunique() != 2 or y.value_counts().min() < 3:
        raise ValueError(
            "Both classes need sufficient observations"
        )

    if X.isna().any().any():
        raise ValueError("Missing feature values detected")

    # Split 80% development and 20% test.
    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    # Split development into 60% train, 20% validation.
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev,
        y_dev,
        test_size=0.25,
        stratify=y_dev,
        random_state=42,
    )

    model = Pipeline(
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
    )

    print("\nTraining Logistic Regression...")
    model.fit(X_train, y_train)

    val_probabilities = model.predict_proba(X_val)[:, 1]

    # Compare candidate thresholds on validation only.
    thresholds = np.arange(0.05, 1.0, 0.05)

    validation_results = [
        calculate_metrics(
            y_val,
            val_probabilities,
            threshold,
        )
        for threshold in thresholds
    ]

    results_df = pd.DataFrame(validation_results)

    print("\nVALIDATION THRESHOLD COMPARISON")
    print(
        results_df[
            [
                "threshold",
                "precision",
                "recall",
                "f1",
                "total_alerts",
            ]
        ].to_string(index=False)
    )

    # Select highest validation F1.
    best = max(
        validation_results,
        key=lambda row: row["f1"],
    )

    best_threshold = best["threshold"]

    print("\nSelected threshold:", best_threshold)

    # Final evaluation on untouched test data.
    test_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    final_metrics = calculate_metrics(
        y_test,
        test_probabilities,
        best_threshold,
    )

    final_metrics["average_precision"] = float(
        average_precision_score(
            y_test,
            test_probabilities,
        )
    )

    print("\nFINAL TEST RESULTS")
    print(json.dumps(final_metrics, indent=2))

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": model,
            "threshold": best_threshold,
            "feature_names": X.columns.tolist(),
        },
        MODEL_DIR / "fraud_model_optimized.joblib",
    )

    (MODEL_DIR / "threshold_metrics.json").write_text(
        json.dumps(
            {
                "validation_selection": best,
                "test_metrics": final_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    results_df.to_csv(
        MODEL_DIR / "threshold_comparison.csv",
        index=False,
    )

    print("\nOptimization completed.")


if __name__ == "__main__":
    optimize_threshold()
