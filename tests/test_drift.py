
import numpy as np
import pandas as pd
import pytest

from fraudlens.drift import (
    calculate_psi,
    classify_drift,
    detect_feature_drift,
)


def test_identical_distribution():

    values = np.arange(1000)

    psi = calculate_psi(
        values,
        values,
    )

    assert psi == pytest.approx(0.0)


def test_stable_distribution():

    rng = np.random.default_rng(42)

    reference = rng.normal(
        0, 1, 10000
    )

    current = rng.normal(
        0, 1, 10000
    )

    assert calculate_psi(
        reference,
        current,
    ) < 0.10


def test_significant_drift():

    rng = np.random.default_rng(42)

    reference = rng.normal(
        0, 1, 10000
    )

    current = rng.normal(
        3, 1, 10000
    )

    assert calculate_psi(
        reference,
        current,
    ) >= 0.25


@pytest.mark.parametrize(
    "psi, expected",
    [
        (0.00, "Stable"),
        (0.09, "Stable"),
        (0.10, "Moderate"),
        (0.24, "Moderate"),
        (0.25, "Significant"),
        (1.50, "Significant"),
    ],
)
def test_drift_classification(
    psi,
    expected,
):

    assert classify_drift(psi) == expected


def test_empty_data():

    with pytest.raises(ValueError):

        calculate_psi(
            [],
            [1, 2, 3],
        )


def test_invalid_values():

    with pytest.raises(ValueError):

        calculate_psi(
            [1, np.nan, 3],
            [1, 2, 3],
        )


def test_constant_reference():

    reference = np.ones(1000)

    current = np.ones(1000) * 5

    assert calculate_psi(
        reference,
        current,
    ) >= 0.25


def test_feature_drift_report():

    rng = np.random.default_rng(42)

    values = rng.normal(
        0, 1, 1000
    )

    reference = pd.DataFrame({
        "V1": values,
        "V2": values,
    })

    current = pd.DataFrame({
        "V1": values + 3,
        "V2": values,
    })

    report = detect_feature_drift(
        reference,
        current,
        features=["V1", "V2"],
    )

    assert len(report) == 2

    assert report.iloc[0]["feature"] == "V1"

    assert report.iloc[0]["status"] == "Significant"


def test_insufficient_samples():

    reference = pd.DataFrame({
        "V1": [1, 2, 3]
    })

    with pytest.raises(ValueError):

        detect_feature_drift(
            reference,
            reference,
            features=["V1"],
        )


def test_missing_feature():

    reference = pd.DataFrame({
        "V1": np.arange(1000)
    })

    with pytest.raises(ValueError):

        detect_feature_drift(
            reference,
            reference,
            features=["V2"],
        )
