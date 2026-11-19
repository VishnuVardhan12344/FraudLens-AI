
import numpy as np
import pandas as pd
import pytest

from fraudlens.transformation import transform_transactions
from fraudlens.validation import DataValidationError


@pytest.fixture
def sample_transactions():
    columns = (
        ["Time"]
        + [f"V{i}" for i in range(1, 29)]
        + ["Amount", "Class"]
    )

    data = {
        column: [0.0, 0.0]
        for column in columns[:-1]
    }

    data["Time"] = [0.0, 90000.0]
    data["Amount"] = [0.0, 100.0]
    data["Class"] = [0, 1]

    return pd.DataFrame(data)


def test_feature_engineering(sample_transactions):
    result = transform_transactions(sample_transactions)

    assert len(result) == 2
    assert len(result.columns) == 35

    assert result["log_amount"].iloc[0] == 0.0
    assert result["log_amount"].iloc[1] == pytest.approx(
        np.log1p(100.0)
    )

    assert result["amount_is_zero"].tolist() == [1, 0]
    assert result["elapsed_day"].tolist() == [0, 1]
    assert result["elapsed_hour"].tolist() == [0, 1]


def test_original_data_is_unchanged(sample_transactions):
    original = sample_transactions.copy(deep=True)

    transform_transactions(sample_transactions)

    pd.testing.assert_frame_equal(
        sample_transactions,
        original,
    )


def test_invalid_input_is_rejected(sample_transactions):
    invalid = sample_transactions.copy()
    invalid.loc[0, "Amount"] = -100.0

    with pytest.raises(
        DataValidationError,
        match="Negative transaction amounts"
    ):
        transform_transactions(invalid)
