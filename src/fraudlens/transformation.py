
import logging

import numpy as np
import pandas as pd

from fraudlens.validation import validate_transactions


logger = logging.getLogger(__name__)


class TransformationError(Exception):
    """Raised when transaction transformation fails."""


def transform_transactions(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Transform validated transaction data.

    This function:
    - Validates incoming records
    - Preserves the original DataFrame
    - Creates deterministic numerical features
    - Returns a new DataFrame
    """

    validate_transactions(df)

    logger.info(
        "Starting transformation for %d transactions",
        len(df),
    )

    transformed = df.copy(deep=True)

    try:
        transformed["log_amount"] = np.log1p(
            transformed["Amount"]
        )

        transformed["amount_is_zero"] = (
            transformed["Amount"] == 0
        ).astype("int8")

        transformed["elapsed_day"] = (
            transformed["Time"] // 86400
        ).astype("int32")

        transformed["elapsed_hour"] = (
            (transformed["Time"] % 86400) // 3600
        ).astype("int8")

    except (ValueError, TypeError, OverflowError) as error:
        raise TransformationError(
            "Transaction feature engineering failed"
        ) from error

    logger.info(
        "Transformation completed: %d rows, %d columns",
        len(transformed),
        len(transformed.columns),
    )

    return transformed
