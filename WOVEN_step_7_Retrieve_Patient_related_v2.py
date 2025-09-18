"""
WOVEN Step 7 — Split by labels
Outputs four subsets:
  1) patient_to_provider == 1
  2) provider_to_provider == 1
  3) provider_to_patient == 1
  4) pt2pr_OR_pr2pt == 1  (patient-to-provider OR provider-to-patient)

Saves each subset to CSV and JSONL under:
  Data/WOVEN/filtered_outputs/checked_results/{CASE_TYPE_FILTER}/
"""

import os
import re
import pandas as pd

# ====== CONFIG ======
CASE_TYPE_FILTER = "Clinical_Question"   # change to the case set you’re working on
INPUT_PATH = f"Data/WOVEN/split_by_case_type/checked_results/{CASE_TYPE_FILTER}_annotated_output.jsonl"
OUT_DIR = f"Data/WOVEN/filtered_outputs/{CASE_TYPE_FILTER}"
os.makedirs(OUT_DIR, exist_ok=True)

# Column names (as used across your pipeline)
PT2PR = "patient-to-provider"
PR2PR = "provider-to-provider"
PR2PT = "provider-to-patient"

def safe_suffix(s: str) -> str:
    s = str(s).strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9._-]", "", s)
    s = re.sub(r"[_-]{2,}", "_", s)
    return f"_{s}" if s else ""

# ====== LOAD ======
if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"Input file not found: {os.path.abspath(INPUT_PATH)}")

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

print(f"Loaded {len(df)} rows from {INPUT_PATH}")

# ====== COERCE FLAGS ======
for col in [PT2PR, PR2PT, PR2PR]:
    if col not in df.columns:
        df[col] = 0
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

# ====== SUBSETS ======
subset_pt2pr = df[df[PT2PR] == 1].copy()
subset_pr2pr = df[df[PR2PR] == 1].copy()
subset_pr2pt = df[df[PR2PT] == 1].copy()
subset_union = df[(df[PT2PR] == 1) | (df[PR2PT] == 1)].copy()

# ====== SAVE ======
SUFFIX = safe_suffix(CASE_TYPE_FILTER)

def save_subset(dframe: pd.DataFrame, stem: str):
    csv_path  = os.path.join(OUT_DIR, f"{stem}{SUFFIX}.csv")
    json_path = os.path.join(OUT_DIR, f"{stem}{SUFFIX}.jsonl")
    dframe.to_csv(csv_path, index=False)
    dframe.to_json(json_path, orient="records", lines=True, force_ascii=False)
    return csv_path, json_path

p1 = save_subset(subset_pt2pr, "patient_to_provider")
p2 = save_subset(subset_pr2pr, "provider_to_provider")
p3 = save_subset(subset_pr2pt, "provider_to_patient")
p4 = save_subset(subset_union, "pt2pr_OR_pr2pt")

# ====== REPORT ======
print("=== Splitting complete ===")
print(f"patient-to-provider == 1 : {len(subset_pt2pr)}  -> {p1[0]} , {p1[1]}")
print(f"provider-to-provider == 1: {len(subset_pr2pr)}  -> {p2[0]} , {p2[1]}")
print(f"provider-to-patient == 1 : {len(subset_pr2pt)}  -> {p3[0]} , {p3[1]}")
print(f"pt2pr OR pr2pt == 1      : {len(subset_union)} -> {p4[0]} , {p4[1]}")
