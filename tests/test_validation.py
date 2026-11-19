
import pandas as pd
import pytest

from fraudlens.validation import (
    DataValidationError,
    validate_transactions,
)


@pytest.fixture
def valid_transactions():
    """Create a small, valid transaction dataset."""

    columns = (
        ["Time"]
        + [f"V{i}" for i in range(1, 29)]
        + ["Amount", "Class"]
    )

    data = {
        column: [0.0, 0.0, 0.0]
        for column in columns[:-1]
    }

    data["Time"] = [10.0, 20.0, 30.0]
    data["Amount"] = [100.0, 500.0, 250.0]
    data["Class"] = [0, 1, 0]

    return pd.DataFrame(data)


def test_valid_transactions(valid_transactions):
    report = validate_transactions(valid_transactions)

    assert report.row_count == 3
    assert report.column_count == 31
    assert report.fraud_count == 1
    assert report.missing_count == 0
    assert report.duplicate_count == 0
