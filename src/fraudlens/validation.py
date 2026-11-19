
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when a dataset violates its data contract."""


@dataclass(frozen=True)
class ValidationReport:
    row_count: int
    column_count: int
    duplicate_count: int
    missing_count: int
    fraud_count: int


def validate_transactions(
    df: pd.DataFrame,
) -> ValidationReport:

    expected_columns = (
        ["Time"]
        + [f"V{i}" for i in range(1, 29)]
        + ["Amount", "Class"]
    )

    missing_columns = set(expected_columns) - set(df.columns)
    unexpected_columns = set(df.columns) - set(expected_columns)

    if missing_columns or unexpected_columns:
        raise DataValidationError(
            f"Schema mismatch. "
            f"Missing: {sorted(missing_columns)}; "
            f"Unexpected: {sorted(unexpected_columns)}"
        )

    if df.empty:
        raise DataValidationError("Dataset is empty")

    missing_count = int(df.isna().sum().sum())

    if missing_count > 0:
        raise DataValidationError(
            f"Found {missing_count} missing values"
        )

    numeric_columns = expected_columns[:-1]

    for column in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise DataValidationError(
                f"Column {column} must be numeric"
            )

    if not np.isfinite(df[numeric_columns].to_numpy()).all():
        raise DataValidationError(
            "Non-finite numerical values detected"
        )

    if (df["Amount"] < 0).any():
        raise DataValidationError(
            "Negative transaction amounts detected"
        )

    if (df["Time"] < 0).any():
        raise DataValidationError(
            "Negative elapsed time detected"
        )

    if not pd.api.types.is_integer_dtype(df["Class"]):
        raise DataValidationError(
            "Class must have an integer data type"
        )

    if not df["Class"].isin([0, 1]).all():
        raise DataValidationError(
            "Invalid fraud classification labels"
        )

    report = ValidationReport(
        row_count=len(df),
        column_count=len(df.columns),
        duplicate_count=int(df.duplicated().sum()),
        missing_count=missing_count,
        fraud_count=int(df["Class"].sum()),
    )

    logger.info("Validation passed: %s", report)

    return report
