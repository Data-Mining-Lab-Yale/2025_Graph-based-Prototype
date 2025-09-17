import pandas as pd
import os
import re

# === File paths ===
input_file = "Data/WOVEN/WOVEN_Pt_Case_Report_cleaned.csv"
output_folder = "Data/WOVEN/split_by_case_type"

# === Load cleaned data ===
df = pd.read_csv(input_file)

# === Create output folder if not exists ===
os.makedirs(output_folder, exist_ok=True)

def safe_slug(s: str) -> str:
    """Make a safe file stem from a case type."""
    s = "Unknown" if pd.isna(s) else str(s).strip()
    s = s.replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9._-]", "", s)      # drop unsafe chars
    s = re.sub(r"[_-]{2,}", "_", s)            # collapse repeats
    return s or "Unknown"

# === Split by Case Type ===
for case_type, subset in df.groupby("Case Type", dropna=False):
    case_label = safe_slug(case_type)

    csv_path  = os.path.join(output_folder, f"{case_label}.csv")
    jsonl_path = os.path.join(output_folder, f"{case_label}.jsonl")  # <- jsonl ext

    # Save CSV and JSONL (preserves all columns incl. new flags)
    subset.to_csv(csv_path, index=False)
    subset.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)

    print(f"Saved {len(subset)} rows to {csv_path} and {jsonl_path}")

print(f"\nAll splits saved to folder: {output_folder}")
