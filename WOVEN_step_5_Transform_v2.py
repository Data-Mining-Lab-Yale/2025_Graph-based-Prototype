"""
JSON/JSONL -> CSV (no fields dropped), with robust loading and is_multiple support
"""

import os
import io
import json
import pandas as pd

# ======= PICK WHICH SET =======
CASE_TYPE_FILTER = "Statement_Billing_or_Insurance_Question"  # change to your set

# ====== INPUT / OUTPUT ======
INPUT_JSON    = f"Data/WOVEN/split_by_case_type/checked_results/{CASE_TYPE_FILTER}_annotated_output.jsonl"
OUTPUT_CSV    = f"Data/WOVEN/split_by_case_type/checked_results/{CASE_TYPE_FILTER}_annotated_human.csv"
OUTPUT_SCHEMA = f"Data/WOVEN/split_by_case_type/checked_results/{CASE_TYPE_FILTER}_annotated_human.schema.json"

# ====== FLAGS (ensure exist, 0/1 ints) ======
FLAG_COLS = [
    "patient-to-provider",
    "provider-to-patient",
    "provider-to-provider",
    "is_telephone_note",
    "is_content",
    "is_multiple",  # keep this new flag
]

def debug_file_info(path: str):
    ap = os.path.abspath(path)
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    print(f"[debug] path={ap}\n        exists={exists} size={size} bytes")
    if exists and size > 0:
        try:
            with open(path, "rb") as f:
                head = f.read(200).decode("utf-8", "replace")
            print(f"[debug] first bytes: {head[:120].replace(chr(10),' \\n ')} ...")
        except Exception as e:
            print(f"[debug] could not preview file: {e}")

def read_any_json(path: str) -> pd.DataFrame:
    """
    Robust reader for .jsonl (lines) or .json (array).
    Uses explicit file objects so pandas never treats the path as literal JSON.
    """
    debug_file_info(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    if os.path.getsize(path) == 0:
        raise ValueError(f"Input file is empty: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        # JSON Lines: one JSON object per line
        with open(path, "rb") as f:
            return pd.read_json(io.BytesIO(f.read()), lines=True)
    elif ext == ".json":
        # Try as JSON Lines first (some people save .json with line-delimited records)
        try:
            with open(path, "rb") as f:
                return pd.read_json(io.BytesIO(f.read()), lines=True)
        except ValueError:
            # Fallback: single JSON array
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return pd.json_normalize(data, sep=".")
            raise ValueError("JSON file is not JSONL or a list/array of objects.")
    else:
        raise ValueError("Expected .jsonl or .json input (CSV not supported in this step).")

def ensure_flags(df: pd.DataFrame) -> pd.DataFrame:
    defaults = {
        "patient-to-provider": 0,
        "provider-to-patient": 0,
        "provider-to-provider": 1,
        "is_telephone_note": 1,
        "is_content": 1,
        "is_multiple": 1,
    }
    for c in FLAG_COLS:
        if c not in df.columns:
            df[c] = defaults[c]
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(defaults[c]).astype(int)
    return df

def stringify_nested(df: pd.DataFrame) -> pd.DataFrame:
    # Convert lists/dicts to JSON strings so CSV keeps them intact
    def _to_str(x):
        if isinstance(x, (list, dict)):
            return json.dumps(x, ensure_ascii=False)
        return x
    obj_cols = df.select_dtypes(include=["object"]).columns
    df[obj_cols] = df[obj_cols].applymap(_to_str)
    return df

def main():
    # Read
    df = read_any_json(INPUT_JSON)
    if df is None or len(df) == 0:
        raise ValueError("No rows loaded from input file.")

    # Flags + stringify nested
    df = ensure_flags(df)
    df = stringify_nested(df)

    # Put common review columns early (keeps ALL columns)
    preferred_front = [c for c in FLAG_COLS + ["Pt. Case Description", "Case Type", "Case Created"] if c in df.columns]
    other_cols = [c for c in df.columns if c not in preferred_front]
    df = df[preferred_front + other_cols]

    # Save CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    # Save schema
    schema = {
        "source_file": INPUT_JSON,
        "rows_written": int(len(df)),
        "columns_count": int(len(df.columns)),
        "columns": list(df.columns),
    }
    with open(OUTPUT_SCHEMA, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    print("=== Export complete ===")
    print(f"Rows written: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"CSV: {OUTPUT_CSV}")
    print(f"Schema: {OUTPUT_SCHEMA}")

if __name__ == "__main__":
    main()
