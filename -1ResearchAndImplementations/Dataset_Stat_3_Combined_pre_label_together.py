# combine_per_label_disagreement.py
# Merge per-label disagreement CSVs (from different semantic reps) side-by-side.

import os
import re
import glob
import pandas as pd

# ====== CONFIG: edit these if you want ======
# Folder to search ('.' = current folder)
SEARCH_DIR = 'Data_for_Evidences/Collected_Ambiguous'

# Pattern(s) of files to merge; you can add more if needed
# Expected filename format examples:
#   sbert_code_per_label_disagreement.csv
#   tfidf_subcode_per_label_disagreement.csv
#   lsa_combined_per_label_disagreement.csv
PATTERNS = [
    "*_code_per_label_disagreement.csv",
    "*_subcode_per_label_disagreement.csv",
    "*_combined_per_label_disagreement.csv",
]

# Output filename templates per level
OUTFILE_TPL = {
    "code": "compare_code_per_label_disagreement.csv",
    "subcode": "compare_subcode_per_label_disagreement.csv",
    "combined": "compare_combined_per_label_disagreement.csv",
}

# Columns expected inside each input CSV
# If your script produces exactly these, you’re good.
EXPECTED_COLS = ["label", "avg_cross_label_rate", "n_items"]

# If any file has slightly different header cases or spaces, we normalize below.
# ===========================================

def normalize_columns(df):
    # Lowercase + strip spaces/underscores for robustness
    rename = {c: re.sub(r'[\s_]+', '_', c.strip().lower()) for c in df.columns}
    df = df.rename(columns=rename)
    # Map to expected names if possible
    mapping = {
        "label": "label",
        "avg_cross_label_rate": "avg_cross_label_rate",
        "avgcrosslabelrate": "avg_cross_label_rate",
        "n_items": "n_items",
        "nitems": "n_items"
    }
    df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
    return df

def parse_method_and_level(fname):
    """
    Extract method and level from filenames like:
      sbert_code_per_label_disagreement.csv
      tfidf_subcode_per_label_disagreement.csv
      lsa_combined_per_label_disagreement.csv
    """
    base = os.path.basename(fname)
    m = re.match(r"(?P<method>.+)_(?P<level>code|subcode|combined)_per_label_disagreement\.csv$", base)
    if not m:
        # Try a looser pattern
        parts = base.split("_")
        method = parts[0]
        # find level token
        level = None
        for tok in parts:
            if tok in ("code", "subcode", "combined"):
                level = tok
                break
        if level is None:
            raise ValueError(f"Cannot parse level from filename: {base}")
        return method, level
    return m.group("method"), m.group("level")

def merge_files(file_list):
    by_level = {"code": [], "subcode": [], "combined": []}
    for f in file_list:
        method, level = parse_method_and_level(f)
        df = pd.read_csv(f)
        df = normalize_columns(df)

        # sanity check
        if "label" not in df.columns:
            raise ValueError(f"'label' column missing in {f}")
        if "avg_cross_label_rate" not in df.columns or "n_items" not in df.columns:
            raise ValueError(f"Expected columns not found in {f}. Got {df.columns.tolist()}")

        # keep only needed columns and suffix with method
        df = df[["label", "avg_cross_label_rate", "n_items"]].copy()
        df = df.rename(columns={
            "avg_cross_label_rate": f"avg_rate_{method}",
            "n_items": f"n_items_{method}"
        })
        by_level[level].append(df)

    # For each level, outer-join all frames on label
    outputs = {}
    for level, frames in by_level.items():
        if not frames:
            continue
        merged = frames[0]
        for df in frames[1:]:
            merged = merged.merge(df, on="label", how="outer")
        # Optional: sort by label
        merged = merged.sort_values("label", kind="stable").reset_index(drop=True)

        # Optional extras: show best method per label
        # Create a column that points to the max avg_rate_* method per row
        rate_cols = [c for c in merged.columns if c.startswith("avg_rate_")]
        if rate_cols:
            merged["best_method"] = merged[rate_cols].idxmax(axis=1).str.replace("avg_rate_", "", regex=False)
            merged["best_rate"] = merged[rate_cols].max(axis=1)

        outputs[level] = merged
    return outputs

def main():
    # Gather files
    files = []
    for pat in PATTERNS:
        files.extend(glob.glob(os.path.join(SEARCH_DIR, pat)))
    files = sorted(set(files))
    if not files:
        raise SystemExit("No matching CSVs found. Check SEARCH_DIR and PATTERNS.")

    print("Merging files:")
    for f in files:
        print(" -", os.path.basename(f))

    outputs = merge_files(files)

    for level, df in outputs.items():
        outname = OUTFILE_TPL.get(level, f"compare_{level}_per_label_disagreement.csv")
        df.to_csv(outname, index=False, encoding="utf-8")
        print(f"[OK] wrote {outname}  (rows={len(df)})")

if __name__ == "__main__":
    main()
