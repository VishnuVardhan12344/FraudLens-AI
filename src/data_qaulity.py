from pathlib import Path
import pandas as pd

project_dir = Path(__file__).resolve().parent.parent
file_path = project_dir / "data" / "raw" / "creditcard.csv"

transactions = pd.read_csv(file_path)
print("data loaded sucessfully")

missing_values = transactions.isnull().sum()
print("missing values:", missing_values)

duplicate_count = transactions.duplicated().sum()

print("Duplicate rows:", duplicate_count)
print("data types:", transactions.dtypes)
negative_amounts = (transactions["Amount"] < 0).sum()

zero_amounts = (transactions["Amount"] == 0).sum()

print("Negative amounts:", negative_amounts)
print("Zero amounts:", zero_amounts)

invalid_labels = (~transactions["Class"].isin([0, 1])).sum()

print("Invalid fraud labels:", invalid_labels)
