# combine_ambiguous_label_pairs.py
# Merge ambiguous_label_pairs_detailed.csv across methods, side by side.

import os
import re
import glob
import pandas as pd

# ====== CONFIG ======
SEARCH_DIR = 'Data_for_Evidences/Collected_Ambiguous'  # folder containing the CSVs

# File patterns per level
PATTERNS = [
    "*_code_ambiguous_label_pairs_detailed.csv",
    "*_subcode_ambiguous_label_pairs_detailed.csv",
    "*_combined_ambiguous_label_pairs_detailed.csv",
]

# Output names per level
OUTFILE_TPL = {
    "code": "compare_code_ambiguous_label_pairs_detailed.csv",
    "subcode": "compare_subcode_ambiguous_label_pairs_detailed.csv",
    "combined": "compare_combined_ambiguous_label_pairs_detailed.csv",
}

# Expected base columns inside each input file
EXPECTED = {
    "label_A", "label_B", "count", "avg_similarity", "n_A", "n_B",
    "ratio_A=count/n_A", "ratio_B=count/n_B"
}
# ====================

def normalize_columns(df):
    # Make headers consistent
    mapping = {
        "label_A": "label_A",
        "label_B": "label_B",
        "count": "count",
        "avg_similarity": "avg_similarity",
        "n_A": "n_A",
        "n_B": "n_B",
        "ratio_A=count/n_A": "ratio_A",
        "ratio_B=count/n_B": "ratio_B",
        # common alternates
        "ratio_A": "ratio_A",
        "ratio_B": "ratio_B",
        "avg-similarity": "avg_similarity",
        "avg_similarity ": "avg_similarity",
    }
    cols = {c: mapping.get(c, c) for c in df.columns}
    df = df.rename(columns=cols)
    return df

def parse_method_and_level(fname):
    """
    Extract method and level from names like:
      sbert_code_ambiguous_label_pairs_detailed.csv
      tfidf_subcode_ambiguous_label_pairs_detailed.csv
      lsa_combined_ambiguous_label_pairs_detailed.csv
    """
    base = os.path.basename(fname)
    m = re.match(r"(?P<method>.+)_(?P<level>code|subcode|combined)_ambiguous_label_pairs_detailed\.csv$", base)
    if not m:
        # fallback: split and search tokens
        parts = base.split("_")
        method = parts[0]
        level = None
        for tok in parts:
            if tok in ("code", "subcode", "combined"):
                level = tok
                break
        if level is None:
            raise ValueError(f"Cannot parse level from: {base}")
        return method, level
    return m.group("method"), m.group("level")

def load_and_prepare(path, method):
    df = pd.read_csv(path)
    df = normalize_columns(df)

    needed = {"label_A","label_B","count","avg_similarity","n_A","n_B","ratio_A","ratio_B"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {os.path.basename(path)}: {missing}")

    # Keep only needed and suffix method
    out = df[["label_A","label_B","count","avg_similarity","n_A","n_B","ratio_A","ratio_B"]].copy()
    out = out.rename(columns={
        "count": f"count_{method}",
        "avg_similarity": f"avg_sim_{method}",
        "n_A": f"n_A_{method}",
        "n_B": f"n_B_{method}",
        "ratio_A": f"ratio_A_{method}",
        "ratio_B": f"ratio_B_{method}",
    })
    return out

def merge_by_level(files):
    buckets = {"code": [], "subcode": [], "combined": []}
    for f in files:
        method, level = parse_method_and_level(f)
        buckets[level].append((method, f))

    outputs = {}
    for level, items in buckets.items():
        if not items:
            continue
        merged = None
        for method, path in items:
            dfm = load_and_prepare(path, method)
            if merged is None:
                merged = dfm
            else:
                merged = merged.merge(dfm, on=["label_A","label_B"], how="outer")

        # Optional: sort for readability
        merged = merged.sort_values(["label_A","label_B"], kind="stable").reset_index(drop=True)

        # Optional helper columns:
        # total conflict count across methods
        count_cols = [c for c in merged.columns if c.startswith("count_")]
        if count_cols:
            merged["count_sum_all_methods"] = merged[count_cols].sum(axis=1, skipna=True)

        outputs[level] = merged
    return outputs

def main():
    files = []
    for pat in PATTERNS:
        files.extend(glob.glob(os.path.join(SEARCH_DIR, pat)))
    files = sorted(set(files))
    if not files:
        raise SystemExit("No matching *_ambiguous_label_pairs_detailed.csv files found.")

    print("Merging:")
    for f in files:
        print(" -", os.path.basename(f))

    outputs = merge_by_level(files)

    for level, df in outputs.items():
        outname = OUTFILE_TPL.get(level, f"compare_{level}_ambiguous_label_pairs_detailed.csv")
        df.to_csv(outname, index=False, encoding="utf-8")
        print(f"[OK] wrote {outname}  rows={len(df)}  cols={len(df.columns)}")

if __name__ == "__main__":
    main()
