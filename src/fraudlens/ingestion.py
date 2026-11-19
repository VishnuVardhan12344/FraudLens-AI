
from pathlib import Path

import pandas as pd


def load_transactions(filepath):
    file_path = Path(filepath)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    return df
