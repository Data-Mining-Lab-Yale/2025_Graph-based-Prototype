"""
Filter annotated dataset for provider-to-patient and patient-to-provider
-----------------------------------------------------------------------
What it does:
- Loads annotated data (.jsonl, .json, or .csv)
- Creates three subsets:
  1) provider_to_patient == 1
  2) patient_to_provider == 1
  3) BOTH == 1 (rows where both flags are 1)
- Saves each subset to CSV and JSONL
- Prints counts

Optional: set CASE_TYPE_FILTER to a string to limit to one case type before filtering.
"""

import os
import pandas as pd


# ======= CASE TYPE FILTER (None to load all) =======
# Examples: None, "Clinical_Question", "Medication", "Other"
CASE_TYPE_FILTER = "Clinical_Question"

# ====== INPUT / OUTPUT ======
INPUT_PATH = f"Data/WOVEN/split_by_case_type/{CASE_TYPE_FILTER}_annotated_output.jsonl"  # set your file
OUT_DIR = f"Data/WOVEN/filtered_outputs/{CASE_TYPE_FILTER}"
os.makedirs(OUT_DIR, exist_ok=True)

# Optional: filter to a single Case Type first (set to None to disable)
CASE_TYPE_FILTER = None  # e.g., "Clinical Question" or "Medication"

# Column names
PT2PR = "patient-to-provider"
PR2PT = "provider-to-patient"
PR2PR = "provider-to-provider"   # not directly used here, but kept for completeness

# ====== LOAD ======
ext = os.path.splitext(INPUT_PATH)[1].lower()
if ext in [".jsonl", ".json"]:
    try:
        df = pd.read_json(INPUT_PATH, lines=True)
    except ValueError:
        df = pd.read_json(INPUT_PATH)
elif ext == ".csv":
    df = pd.read_csv(INPUT_PATH)
else:
    raise ValueError("INPUT_PATH must be .jsonl, .json, or .csv")

# ====== BASIC CHECKS ======
for col in [PT2PR, PR2PT]:
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in file!")

# Coerce flags to integers (0/1) in case they are strings
df[PT2PR] = pd.to_numeric(df[PT2PR], errors="coerce").fillna(0).astype(int)
df[PR2PT] = pd.to_numeric(df[PR2PT], errors="coerce").fillna(0).astype(int)

# Optional: filter by case type before splitting
if CASE_TYPE_FILTER is not None and "Case Type" in df.columns:
    df = df[df["Case Type"].astype(str) == str(CASE_TYPE_FILTER)].copy()

# ====== SUBSETS ======
subset_pr2pt = df[df[PR2PT] == 1].copy()
subset_pt2pr = df[df[PT2PR] == 1].copy()
subset_both  = df[(df[PR2PT] == 1) & (df[PT2PR] == 1)].copy()

# ====== SAVE HELPERS ======
def save_subset(dframe: pd.DataFrame, stem: str, case_filter: str = None):
    # Prepare safe suffix for filenames
    suffix = f"_{case_filter.replace(' ', '_')}" if case_filter else ""
    csv_path  = os.path.join(OUT_DIR, f"{stem}{suffix}.csv")
    json_path = os.path.join(OUT_DIR, f"{stem}{suffix}.jsonl")
    dframe.to_csv(csv_path, index=False)
    dframe.to_json(json_path, orient="records", lines=True, force_ascii=False)
    return csv_path, json_path

# ====== SAVE ======
p1 = save_subset(subset_pr2pt, "provider_to_patient", CASE_TYPE_FILTER)
p2 = save_subset(subset_pt2pr, "patient_to_provider", CASE_TYPE_FILTER)
p3 = save_subset(subset_both,  "both_pt2pr_and_pr2pt", CASE_TYPE_FILTER)

# ====== REPORT ======
print("=== Filtering complete ===")
if CASE_TYPE_FILTER is not None:
    print(f"Case Type filter applied: {CASE_TYPE_FILTER}")
print(f"Total rows in input: {len(df)}")
print(f"provider-to-patient == 1: {len(subset_pr2pt)}  -> {p1[0]} , {p1[1]}")
print(f"patient-to-provider == 1: {len(subset_pt2pr)}  -> {p2[0]} , {p2[1]}")
print(f"both flags == 1        : {len(subset_both)}   -> {p3[0]} , {p3[1]}")
