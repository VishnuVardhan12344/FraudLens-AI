from pathlib import Path
import pandas as pd

project_dir = Path(__file__).resolve().parent.parent

raw_file = project_dir / "data" / "raw" / "creditcard.csv"

transactions = pd.read_csv(raw_file)

print("Dataset loaded successfully")
print("Original rows:", len(transactions))

cleaned_transactions = transactions.copy()

missing_count = cleaned_transactions.isnull().sum().sum()
#First .#sum() counts missing values in each column.
#Second .sum() adds those column counts together to get the total.
print("Total missing values:", missing_count)

if missing_count > 0:
    raise ValueError("Missing values detected. Investigate before cleaning.")

valid_labels = cleaned_transactions["Class"].isin([0, 1])

invalid_count = (~valid_labels).sum()

print("Invalid fraud labels:", invalid_count)

if invalid_count > 0:
    raise ValueError("Invalid fraud labels found.")

negative_count = (cleaned_transactions["Amount"] < 0).sum()

print("Negative amounts:", negative_count)

if negative_count > 0:
    raise ValueError("Negative amounts detected. Investigate first.")

duplicate_count = cleaned_transactions.duplicated().sum()

print("Identical rows detected:", duplicate_count)
print("Duplicates retained for investigation.")

processed_dir = project_dir / "data" / "processed"

processed_dir.mkdir(parents=True, exist_ok=True)

output_file = processed_dir / "creditcard_cleaned.csv"

cleaned_transactions.to_csv(output_file, index=False)

print("Processed dataset saved:", output_file)