
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fraudlens.drift import FEATURE_NAMES


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "data" / "monitoring"


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        required=True,
    )

    args = parser.parse_args()

    df = pd.read_csv(args.data)

    df = df[FEATURE_NAMES].dropna()

    if len(df) < 20000:
        raise ValueError(
            "At least 20,000 records required"
        )

    sampled = df.sample(
        n=20000,
        random_state=42,
    ).reset_index(drop=True)

    reference = sampled.iloc[:10000].copy()

    current = sampled.iloc[10000:].copy()

    drifted = current.copy()

    rng = np.random.default_rng(42)

    # Simulate a change in transaction behavior

    drifted["Amount"] *= 3

    drifted["V1"] += 2.0

    drifted["V2"] -= 1.5

    drifted["V3"] += rng.normal(
        loc=1.0,
        scale=0.5,
        size=len(drifted),
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference.to_csv(
        OUTPUT_DIR / "reference.csv",
        index=False,
    )

    current.to_csv(
        OUTPUT_DIR / "current_stable.csv",
        index=False,
    )

    drifted.to_csv(
        OUTPUT_DIR / "current_drifted.csv",
        index=False,
    )

    print("PASS: Reference data created")
    print("PASS: Stable current data created")
    print("PASS: Synthetic drift data created")

    print(f"Reference records: {len(reference)}")
    print(f"Current records: {len(current)}")


if __name__ == "__main__":
    main()
