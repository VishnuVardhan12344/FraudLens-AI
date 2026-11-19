from pathlib import Path
import pandas as pd
project_dir = Path(__file__).resolve().parent.parent\
##This line finds the full location of the current Python file, 
## moves up two folder levels to reach the main project folder, and stores that folder's path in the variable project_dir.
#project_dir	Variable that stores the main project folder's location
#=	Assignment operator; stores the result on the right in the variable on the left
#Path()	Creates a Path object to work with file and folder locations
#__file__	Represents the location of the currently running Python script
#.resolve()	Converts the location to an absolute path and resolves symbolic links
#First .parent	Moves one level up from load_data.py to the src folder
#Second .parent	Moves another level up from src to FRAUDLENS_AI
file_path = project_dir / "data" / "raw" / "creditcard.csv"
transactions = pd.read_csv(file_path)
print(transactions.head())
print("===== FRAUDLENS AI =====")

print(transactions.head())

print("Dataset Shape:", transactions.shape)

print("Columns:", transactions.columns.tolist())
print(transactions.head(10))
