import pandas as pd
from pathlib import Path

directory_path = Path(r"C:\Users\JacopoBinati\OneDrive - Venionaire Capital\Desktop\crunchbase\raw files\multiples")

file_names = [f"multiples-{i}.csv" for i in range(1, 11)]
dataframes = [pd.read_csv(directory_path / file_name) for file_name in file_names]

df_combined = pd.concat(dataframes, ignore_index=True)
output_path = directory_path / "multiples_combined.csv"
df_combined.to_csv(output_path, index=False)

columns_to_drop = [
    "Last Funding Amount", 
    "Last Equity Funding Amount Currency", 
    "Last Equity Funding Type",
    "Last Equity Funding Amount", 
    "Total Equity Funding Amount", 
    "Total Equity Funding Amount Currency",  
    "Total Funding Amount", 
    "Total Funding Amount Currency",
    "Valuation at IPO Currency",
    "Valuation at IPO (in USD)",
    "Valuation at IPO"
]
df_combined = df_combined.drop(columns=columns_to_drop)
df_combined.to_csv(output_path, index=False)