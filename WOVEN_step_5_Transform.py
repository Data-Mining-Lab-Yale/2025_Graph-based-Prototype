"""
JSON/JSONL -> CSV (no fields dropped)
- Preserves ALL keys across all records (union of keys).
- Serializes nested fields (lists/dicts) to JSON strings.
- Ensures flag columns exist and are int (0/1).
- Writes a sidecar schema JSON listing the exact columns written.
- Prints a quick sanity report (rows/cols) so you can confirm parity.

Usage:
1) Set INPUT_JSON below to your annotated JSON or JSONL (line-delimited) file.
2) Run:  python json_to_csv_preserve_all.py
"""

import os
import json
import pandas as pd

# ======= CASE TYPE FILTER (None to load all) =======
# Examples: None, "Clinical_Question", "Medication", "Other"
# CASE_TYPE_FILTER = "Clinical_Question"
CASE_STEM = "Statement_Billing_or_Insurance_Question"  


# ====== INPUT / OUTPUT ======
INPUT_JSON = f"Data/WOVEN/split_by_case_type/{CASE_TYPE_FILTER}_annotated_output.jsonl"   # your annotated JSON/JSONL file
OUTPUT_CSV  = f"Data/WOVEN/split_by_case_type/{CASE_TYPE_FILTER}_annotated_human.csv"     # human-readable CSV
OUTPUT_SCHEMA = f"Data/WOVEN/split_by_case_type/{CASE_TYPE_FILTER}_annotated_human.schema.json"  # columns written (for verification)

# ====== FLAGS we expect (ensure they exist, 0/1 ints) ======
FLAG_COLS = [
    "patient-to-provider",
    "provider-to-patient",
    "provider-to-provider",
    "is_telephone_note",
    "is_content",
]

def read_any_json(path: str) -> pd.DataFrame:
    """
    Reads either JSON Lines (.jsonl) or a single JSON array (.json).
    Falls back clearly if format is unexpected.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        return pd.read_json(path, lines=True)
    elif ext == ".json":
        # Try as JSON Lines first (robust), else as array
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            # Load array and normalize to rows
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return pd.json_normalize(data, sep=".")
            raise ValueError("JSON file is not JSONL or a list/array of objects.")
    else:
        raise ValueError("Expected .json or .jsonl input.")

def ensure_flags(df: pd.DataFrame) -> pd.DataFrame:
    for c in FLAG_COLS:
        if c not in df.columns:
            df[c] = 0
        # coerce to 0/1 int if possible
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df

def stringify_nested(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert lists/dicts to JSON strings so they are not mangled in CSV.
    Leave scalars and NaN as-is.
    """
    def _to_str(x):
        if isinstance(x, (list, dict)):
            return json.dumps(x, ensure_ascii=False)
        return x
    # Only apply to object dtype columns to keep it fast
    obj_cols = df.select_dtypes(include=["object"]).columns
    df[obj_cols] = df[obj_cols].applymap(_to_str)
    return df

def main():
    # Read
    df = read_any_json(INPUT_JSON)

    # Make sure we didn’t accidentally drop nested fields
    # (json_normalize already flattens dicts with key paths if loaded via array branch)
    # For JSONL, records are typically flat; if not, stringify below preserves content.
    df = ensure_flags(df)
    df = stringify_nested(df)

    # (Optional) Put common review columns early; **does not drop any**
    preferred_front = [c for c in FLAG_COLS + ["Pt. Case Description", "Case Type", "Case Created"] if c in df.columns]
    other_cols = [c for c in df.columns if c not in preferred_front]
    ordered_cols = preferred_front + other_cols
    df = df[ordered_cols]

    # Write CSV (quotes handled by pandas; multiline fields preserved)
    df.to_csv(OUTPUT_CSV, index=False)

    # Write schema for verification
    schema = {
        "source_file": INPUT_JSON,
        "rows_written": int(len(df)),
        "columns_count": int(len(df.columns)),
        "columns": list(df.columns),
    }
    with open(OUTPUT_SCHEMA, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    # Console sanity report
    print("=== Export complete ===")
    print(f"Rows written: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"CSV: {OUTPUT_CSV}")
    print(f"Schema: {OUTPUT_SCHEMA}")

if __name__ == "__main__":
    main()
