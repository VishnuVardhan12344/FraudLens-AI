
import logging
from pathlib import Path

from fraudlens.pipeline import run_pipeline


logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)


PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_PATH = (
    PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "creditcard_features.parquet"
)


if __name__ == "__main__":
    result = run_pipeline(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
    )

    print("\nFraudLens Pipeline Result")
    print(result)
