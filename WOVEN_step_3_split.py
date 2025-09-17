import pandas as pd
import os

# === File paths ===
input_file = "Data/WOVEN/WOVEN_Pt_Case_Report_cleaned.csv"
output_folder = "Data/WOVEN/split_by_case_type"

# === Load cleaned data ===
df = pd.read_csv(input_file)

# === Create output folder if not exists ===
os.makedirs(output_folder, exist_ok=True)

# === Split by Case Type ===
for case_type, subset in df.groupby("Case Type", dropna=False):
    # Handle missing/NaN case type
    case_label = "Unknown" if pd.isna(case_type) else str(case_type).replace(" ", "_")
    
    csv_path = os.path.join(output_folder, f"{case_label}.csv")
    json_path = os.path.join(output_folder, f"{case_label}.json")
    
    # Save CSV and JSON
    subset.to_csv(csv_path, index=False)
    subset.to_json(json_path, orient="records", lines=True)
    
    print(f"Saved {len(subset)} rows to {csv_path} and {json_path}")

print(f"\nAll splits saved to folder: {output_folder}")
