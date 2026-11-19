#Time	Seconds elapsed since the first transaction
#V1 to V28	Anonymized numerical features, mostly produced through PCA
#Amount	Transaction amount
#Class	Actual fraud label: 0 = legitimate, 1 = fraud
from pathlib import Path
import pandas as pd

project_dir = Path(__file__).resolve().parent.parent
file_path = project_dir / "data" / "raw" / "creditcard.csv"

transactions = pd.read_csv(file_path)

print("FraudLens AI - Transaction Analysis")
total_transactions = len(transactions)
print("TOTAL TRANSACTIOS :",total_transactions)
fraud_transactions = transactions[transactions["Class"] == 1]
fraud_count = len(fraud_transactions)
print("total fraud counts:", fraud_count)
legitmate_transactions = transactions[transactions["Class"] == 0]
legitmate_count = len(legitmate_transactions)
print("total legitmate counts:", legitmate_count)

fraud_percentage = (fraud_count/total_transactions)*100
print(f"fraud percentage:{fraud_percentage:.4f}%")
legitmate_percentage = (legitmate_count/total_transactions)*100
print(f"legitmate percentage:{legitmate_percentage:.4f}%")

largest_amount = transactions["Amount"].max()
smallest_amount = transactions["Amount"].min()
average_amount = transactions["Amount"].mean()
print(f"largest_amount: ${largest_amount}")
print(f"smallest_amount: ${smallest_amount}")
print(f"average_amount: ${average_amount}")