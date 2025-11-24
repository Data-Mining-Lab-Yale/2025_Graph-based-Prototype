# combine_nn_conflict_samples.py
# Merge example files from multiple semantic representations and levels.

import os
import re
import glob
import hashlib
import pandas as pd

# ========== CONFIG ==========
SEARCH_DIR = "Data_for_Evidences/Collected_Ambiguous"  # folder with your CSVs

# Patterns to collect by level. Add or remove as needed.
PATTERNS = [
    "*_code_nn_conflict_samples.csv",
    "*_subcode_nn_conflict_samples.csv",
    "*_combined_nn_conflict_samples.csv",
]

# Output files
OUT_COMBINED_LONG_TPL = "compare_{level}_nn_conflict_samples_LONG.csv"
OUT_TOPK_TPL          = "compare_{level}_nn_conflict_samples_top{K}.csv"
OUT_OVERLAP_TPL       = "compare_{level}_nn_conflict_samples_OVERLAP.csv"
OUT_COUNTS_TPL        = "compare_{level}_nn_conflict_samples_COUNTS.csv"

TOP_K = 5  # per label pair and method, keep the top K by similarity

# Expected columns in each input examples file
# If names differ slightly, we normalize below
EXPECTED_COLS = [
    "anchor_text", "anchor_label", "neighbor_text", "neighbor_label", "similarity"
]
# ============================

def normalize_columns(df):
    # normalize common variants of column names
    mapping = {}
    cols = {c.strip(): c.strip().lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=cols)
    # map to canonical
    canon = {
        "anchor_text": "anchor_text",
        "anchorlabel": "anchor_label",
        "anchor_label": "anchor_label",
        "neighbor_text": "neighbor_text",
        "neighborlabel": "neighbor_label",
        "neighbor_label": "neighbor_label",
        "similarity": "similarity",
    }
    mapping = {c: canon.get(c, c) for c in df.columns}
    df = df.rename(columns=mapping)
    return df

def parse_method_and_level(fname):
    base = os.path.basename(fname)
    m = re.match(r"(?P<method>.+)_(?P<level>code|subcode|combined)_nn_conflict_samples\.csv$", base)
    if not m:
        parts = base.split("_")
        method = parts[0]
        level = None
        for tok in parts:
            if tok in ("code", "subcode", "combined"):
                level = tok
                break
        if level is None:
            raise ValueError(f"Cannot parse level from filename: {base}")
        return method, level
    return m.group("method"), m.group("level")

def hash_pair(a_text, a_label, n_text, n_label):
    # stable identifier to detect exact same example across methods
    h = hashlib.sha1()
    key = f"{a_label}||{a_text}>>{n_label}||{n_text}"
    h.update(key.encode("utf-8"))
    return h.hexdigest()

def load_examples(path, method, level):
    df = pd.read_csv(path)
    df = normalize_columns(df)
    missing = set(EXPECTED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {os.path.basename(path)}: {missing}")
    df = df[EXPECTED_COLS].copy()
    df["method"] = method
    df["level"] = level
    # useful derived fields
    df["pair_key"] = df["anchor_label"].astype(str) + " :: " + df["neighbor_label"].astype(str)
    df["example_hash"] = [
        hash_pair(a, al, n, nl)
        for a, al, n, nl in zip(df["anchor_text"], df["anchor_label"], df["neighbor_text"], df["neighbor_label"])
    ]
    return df

def collect_files():
    files = []
    for pat in PATTERNS:
        files.extend(glob.glob(os.path.join(SEARCH_DIR, pat)))
    files = sorted(set(files))
    if not files:
        raise SystemExit("No matching *_nn_conflict_samples.csv found. Check SEARCH_DIR and PATTERNS.")
    return files

def combine_by_level(files):
    buckets = {"code": [], "subcode": [], "combined": []}
    for f in files:
        method, level = parse_method_and_level(f)
        buckets[level].append((method, f))

    outputs = {}
    for level, items in buckets.items():
        if not items:
            continue
        frames = []
        for method, path in items:
            frames.append(load_examples(path, method, level))
        big = pd.concat(frames, ignore_index=True)
        outputs[level] = big
    return outputs

def make_topk_per_pair_per_method(df_level, top_k=TOP_K):
    # sort by similarity desc within each (pair_key, method), then pick head(top_k)
    df_sorted = df_level.sort_values(["pair_key", "method", "similarity"], ascending=[True, True, False])
    out = df_sorted.groupby(["pair_key", "method"], as_index=False).head(top_k)
    # Optional: make reading easier
    out = out.sort_values(["pair_key", "method", "similarity"], ascending=[True, True, False]).reset_index(drop=True)
    return out

def make_overlap(df_level):
    # find example_hash that appear in more than one method
    counts = df_level.groupby("example_hash")["method"].nunique().reset_index(name="n_methods")
    dup_hashes = set(counts.loc[counts["n_methods"] > 1, "example_hash"])
    overlap = df_level[df_level["example_hash"].isin(dup_hashes)].copy()
    # sort for readability
    overlap = overlap.sort_values(["example_hash", "method", "similarity"], ascending=[True, True, False]).reset_index(drop=True)
    return overlap

def make_counts_table(df_level):
    # count examples per (label pair, method)
    tab = df_level.groupby(["pair_key", "method"], as_index=False).size().rename(columns={"size": "n_examples"})
    # pivot so columns become methods
    wide = tab.pivot(index="pair_key", columns="method", values="n_examples").fillna(0).astype(int)
    wide = wide.reset_index()
    # also total across methods
    method_cols = [c for c in wide.columns if c != "pair_key"]
    wide["total_all_methods"] = wide[method_cols].sum(axis=1)
    return wide

def main():
    files = collect_files()
    print("Merging example files:")
    for f in files:
        print(" -", os.path.basename(f))

    by_level = combine_by_level(files)

    for level, df_level in by_level.items():
        # 1) long combined file
        long_name = OUT_COMBINED_LONG_TPL.format(level=level)
        df_level.to_csv(long_name, index=False, encoding="utf-8")
        print(f"[OK] wrote {long_name}  rows={len(df_level)}  cols={len(df_level.columns)}")

        # 2) per pair and method top-K examples by similarity
        topk_df = make_topk_per_pair_per_method(df_level, top_k=TOP_K)
        topk_name = OUT_TOPK_TPL.format(level=level, K=TOP_K)
        topk_df.to_csv(topk_name, index=False, encoding="utf-8")
        print(f"[OK] wrote {topk_name}  rows={len(topk_df)}")

        # 3) overlap examples that appear in multiple methods
        overlap_df = make_overlap(df_level)
        overlap_name = OUT_OVERLAP_TPL.format(level=level)
        overlap_df.to_csv(overlap_name, index=False, encoding="utf-8")
        print(f"[OK] wrote {overlap_name}  rows={len(overlap_df)}")

        # 4) counts table per label pair and method
        counts_df = make_counts_table(df_level)
        counts_name = OUT_COUNTS_TPL.format(level=level)
        counts_df.to_csv(counts_name, index=False, encoding="utf-8")
        print(f"[OK] wrote {counts_name}  rows={len(counts_df)}")

if __name__ == "__main__":
    main()
