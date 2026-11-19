
import hashlib
import json
import logging
import os
import tempfile

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fraudlens.ingestion import load_transactions
from fraudlens.validation import validate_transactions
from fraudlens.transformation import transform_transactions


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    input_sha256: str
    row_count: int
    fraud_count: int
    duplicate_count: int
    output_path: str
    completed_at: str


def calculate_sha256(file_path: Path) -> str:
    """Calculate a streaming SHA-256 checksum."""

    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def atomic_write_parquet(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write Parquet without exposing partial output."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".parquet",
            prefix=".fraudlens-",
            dir=output_path.parent,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)

        df.to_parquet(
            temp_path,
            engine="pyarrow",
            index=False,
            compression="snappy",
        )

        verified = pd.read_parquet(temp_path)

        if verified.shape != df.shape:
            raise RuntimeError(
                "Parquet verification failed: shape mismatch"
            )

        if list(verified.columns) != list(df.columns):
            raise RuntimeError(
                "Parquet verification failed: schema mismatch"
            )

        os.replace(temp_path, output_path)

        logger.info(
            "Parquet committed to %s",
            output_path,
        )

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def run_pipeline(
    input_path: Path,
    output_path: Path,
) -> PipelineResult:
    """Run ingestion, validation and transformation."""

    input_path = Path(input_path)
    output_path = Path(output_path)

    logger.info("Starting FraudLens pipeline")

    input_sha256 = calculate_sha256(input_path)

    transactions = load_transactions(input_path)

    validation_report = validate_transactions(
        transactions
    )

    transformed = transform_transactions(
        transactions
    )

    if len(transformed) != validation_report.row_count:
        raise RuntimeError(
            "Row count changed unexpectedly"
        )

    atomic_write_parquet(
        transformed,
        output_path,
    )

    result = PipelineResult(
        input_sha256=input_sha256,
        row_count=validation_report.row_count,
        fraud_count=validation_report.fraud_count,
        duplicate_count=validation_report.duplicate_count,
        output_path=str(output_path),
        completed_at=datetime.now(
            timezone.utc
        ).isoformat(),
    )

    metadata_path = output_path.with_suffix(
        ".metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            asdict(result),
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("Pipeline completed successfully")

    return result
