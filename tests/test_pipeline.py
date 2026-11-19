
import json

import pandas as pd

from fraudlens.pipeline import run_pipeline


def test_end_to_end_pipeline(tmp_path):
    columns = (
        ["Time"]
        + [f"V{i}" for i in range(1, 29)]
        + ["Amount", "Class"]
    )

    data = {
        column: [0.0, 0.0]
        for column in columns[:-1]
    }

    data["Time"] = [10.0, 90000.0]
    data["Amount"] = [100.0, 500.0]
    data["Class"] = [0, 1]

    input_df = pd.DataFrame(data)

    input_path = tmp_path / "transactions.csv"
    output_path = tmp_path / "features.parquet"

    input_df.to_csv(input_path, index=False)

    result = run_pipeline(
        input_path=input_path,
        output_path=output_path,
    )

    assert result.row_count == 2
    assert result.fraud_count == 1
    assert output_path.exists()

    processed = pd.read_parquet(output_path)

    assert processed.shape == (2, 35)
    assert "log_amount" in processed.columns

    metadata_path = output_path.with_suffix(
        ".metadata.json"
    )

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    assert metadata["row_count"] == 2
    assert metadata["fraud_count"] == 1
    assert len(metadata["input_sha256"]) == 64
