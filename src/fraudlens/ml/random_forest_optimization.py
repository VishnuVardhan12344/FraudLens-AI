
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = (
    ROOT / "data" / "processed"
    / "creditcard_features.parquet"
)

MODEL_DIR = ROOT / "models"


def evaluate(y_true, probabilities, threshold):

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
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "total_alerts": int(tp + fp),
    }


def optimize_random_forest():

    df = pd.read_parquet(DATA_PATH)

    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    # Reserve the final test set.
    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    # Split development into training and validation.
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev,
        y_dev,
        test_size=0.25,
        stratify=y_dev,
        random_state=42,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    val_probabilities = model.predict_proba(
        X_val
    )[:, 1]

    thresholds = np.arange(
        0.05, 1.0, 0.05
    )

    results = []

    for threshold in thresholds:
        metrics = evaluate(
            y_val,
            val_probabilities,
            threshold,
        )
        results.append(metrics)

    comparison = pd.DataFrame(results)

    print("\nVALIDATION THRESHOLD COMPARISON")
    print(
        comparison[
            [
                "threshold",
                "precision",
                "recall",
                "f1",
                "total_alerts",
            ]
        ].to_string(index=False)
    )

    # Choose the threshold using validation only.
    best = max(
        results,
        key=lambda row: row["f1"],
    )

    best_threshold = best["threshold"]

    print("\nBest validation threshold:")
    print(best_threshold)

    # Evaluate the selected threshold on test data.
    test_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    final_results = evaluate(
        y_test,
        test_probabilities,
        best_threshold,
    )

    final_results["average_precision"] = float(
        average_precision_score(
            y_test,
            test_probabilities,
        )
    )

    print("\nFINAL TEST RESULTS")
    print(json.dumps(final_results, indent=2))

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
        MODEL_DIR / "random_forest_optimized.joblib",
    )

    comparison.to_csv(
        MODEL_DIR / "rf_threshold_comparison.csv",
        index=False,
    )

    (MODEL_DIR / "rf_threshold_metrics.json").write_text(
        json.dumps(
            {
                "validation_selection": best,
                "test_metrics": final_results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nRandom Forest optimization complete!")


if __name__ == "__main__":
    optimize_random_forest()
