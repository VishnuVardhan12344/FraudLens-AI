
import numpy as np
import pandas as pd


FEATURE_NAMES = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)


def calculate_psi(
    reference,
    current,
    bins=10,
    epsilon=1e-6,
):

    reference = np.asarray(
        reference, dtype=float
    )

    current = np.asarray(
        current, dtype=float
    )

    if reference.ndim != 1 or current.ndim != 1:
        raise ValueError("Expected one-dimensional data")

    if len(reference) == 0 or len(current) == 0:
        raise ValueError("Input data cannot be empty")

    if not np.isfinite(reference).all():
        raise ValueError("Invalid reference values")

    if not np.isfinite(current).all():
        raise ValueError("Invalid current values")

    if bins < 2:
        raise ValueError("At least two bins required")

    quantiles = np.linspace(0, 1, bins + 1)

    boundaries = np.unique(
        np.quantile(reference, quantiles)
    )

    if len(boundaries) == 1:

        value = boundaries[0]

        boundaries = np.array([
            -np.inf,
            value,
            np.nextafter(value, np.inf),
            np.inf,
        ])

    else:

        boundaries[0] = -np.inf
        boundaries[-1] = np.inf

    reference_counts = np.histogram(
        reference,
        bins=boundaries,
    )[0]

    current_counts = np.histogram(
        current,
        bins=boundaries,
    )[0]

    reference_ratio = (
        reference_counts / len(reference)
    )

    current_ratio = (
        current_counts / len(current)
    )

    reference_ratio = np.maximum(
        reference_ratio, epsilon
    )

    current_ratio = np.maximum(
        current_ratio, epsilon
    )

    reference_ratio /= reference_ratio.sum()
    current_ratio /= current_ratio.sum()

    psi = np.sum(
        (current_ratio - reference_ratio)
        * np.log(
            current_ratio / reference_ratio
        )
    )

    return float(max(0.0, psi))


def classify_drift(psi):

    if psi < 0.10:
        return "Stable"

    if psi < 0.25:
        return "Moderate"

    return "Significant"


def detect_feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features=None,
    min_samples=500,
):

    if features is None:
        features = FEATURE_NAMES

    if (
        len(reference) < min_samples
        or len(current) < min_samples
    ):
        raise ValueError(
            "Insufficient data for drift detection"
        )

    results = []

    for feature in features:

        if (
            feature not in reference.columns
            or feature not in current.columns
        ):
            raise ValueError(
                f"Missing feature: {feature}"
            )

        psi = calculate_psi(
            reference[feature],
            current[feature],
        )

        results.append({
            "feature": feature,
            "psi": round(psi, 6),
            "status": classify_drift(psi),
        })

    report = pd.DataFrame(results)

    return report.sort_values(
        "psi",
        ascending=False,
    ).reset_index(drop=True)

