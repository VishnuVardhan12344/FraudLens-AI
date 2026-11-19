
from pathlib import Path

import httpx
import pandas as pd


ROOT = Path(__file__).resolve().parent

DATA_PATH = (
    ROOT / "data" / "processed"
    / "creditcard_features.parquet"
)

df = pd.read_parquet(DATA_PATH)

columns = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)

for actual_class in [0, 1]:

    row = df[df["Class"] == actual_class].iloc[0]

    payload = {
        "features": {
            col: float(row[col])
            for col in columns
        }
    }

    response = httpx.post(
        "http://127.0.0.1:8000/predict",
        json=payload,
        timeout=30.0,
    )

    response.raise_for_status()
    result = response.json()

    print("\n-------------------------")
    print("Actual class:", actual_class)
    print("Predicted class:", result["prediction"])
    print("Fraud probability:", result["fraud_probability"])
    print("Threshold:", result["threshold"])
    print("-------------------------")
