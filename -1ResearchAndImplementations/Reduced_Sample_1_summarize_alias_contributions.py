#!/usr/bin/env python3
"""
Summarize interactional alias contributions from toyset files.

Quick use:
1) Edit the CONFIG block below (paths and field names).
2) Run:  python Reduced_Sample_1_summarize_alias_contributions.py

Outputs:
- <OUT_PREFIX>_alias_summary.csv
- <OUT_PREFIX>_subcode_breakdown.csv
- <OUT_PREFIX>_alias_summary.json
- <OUT_PREFIX>_alias_totals.png
"""

# =========================
# CONFIG: edit these values
# =========================

TOYSET_JSON = "toysets_out/interactional_toyset_raw.json"          # <— set your toyset path here
ALIAS_MAP   = "toysets_out/alias_verification_interactional.csv"    # <— set your alias map CSV/JSON path here
LABEL_FIELD = "selected_labels"                          # field in toyset with target aliases
RAW_FIELD   = ""                               # field with original subcodes; set "" if absent
OUT_PREFIX  = "toysets_out/toy_interactional_alias_summary"              # prefix for output files

# No need to change anything below
import json, os
from collections import defaultdict, Counter
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt

RAW_COL_CANDIDATES   = ["raw_label", "source_label", "subcode", "original_label", "raw"]
ALIAS_COL_CANDIDATES = ["target", "alias", "target_alias", "mapped_to", "selected_alias","target_label"]

def read_alias_map(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict) and "rows" in data:
            df = pd.DataFrame(data["rows"])
        else:
            df = pd.json_normalize(data)
    else:
        raise ValueError(f"Unsupported alias map extension: {ext}")
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def find_column(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find any of {candidates} in columns {list(df.columns)}")

# def load_toyset(path: str) -> List[dict]:
#     with open(path, "r", encoding="utf-8") as f:
#         data = json.load(f)
#     if isinstance(data, dict) and "data" in data:
#         data = data["data"]
#     if not isinstance(data, list):
#         raise ValueError("Toyset JSON must be a list or a dict with key 'data'.")
#     return data
def load_toyset(path: str) -> list[dict]:
    """
    Load toyset records from a variety of JSON shapes:
      - List[...]                                       -> return as-is
      - {"data":[...]}                                  -> return ["data"]
      - {"records":[...]}, {"rows":[...]}, {"items":[...]} -> return that list
      - {"train":[...], "dev":[...], "test":[...]}      -> concat all split lists
      - {"id1": {...}, "id2": {...}, ...}               -> return list(dict.values())
      - JSONL (.jsonl): one JSON object per line        -> return list of lines
    """
    import os, json

    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        # Newline-delimited JSON
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Case 1: already a list
    if isinstance(data, list):
        return data

    # Case 2: common container keys
    if isinstance(data, dict):
        for key in ["data", "records", "rows", "items"]:
            if key in data and isinstance(data[key], list):
                return data[key]

        # Case 3: split containers
        split_keys = [k for k in ["train", "valid", "val", "dev", "test"] if k in data and isinstance(data[k], list)]
        if split_keys:
            out = []
            for k in split_keys:
                out.extend(data[k])
            return out

        # Case 4: dict-of-objects -> take values
        if all(isinstance(v, dict) for v in data.values()):
            return list(data.values())

    # If we get here, show a helpful sample of the structure
    raise ValueError(
        "Unsupported toyset JSON shape. Top-level type: "
        f"{type(data).__name__}. Keys: {list(data)[:10] if isinstance(data, dict) else 'N/A'}"
    )


def extract_labels(rec: dict, label_field: str) -> List[str]:
    if not label_field:
        return []
    val = rec.get(label_field, [])
    if isinstance(val, str):
        return [val]
    if isinstance(val, list):
        out = []
        for v in val:
            if isinstance(v, str):
                out.append(v)
            elif isinstance(v, dict) and "label" in v:
                out.append(str(v["label"]))
        return out
    return []

def extract_raw_labels(rec: dict, raw_field: str) -> List[str]:
    if not raw_field:
        return []
    val = rec.get(raw_field, [])
    if isinstance(val, str):
        return [val]
    if isinstance(val, list):
        out = []
        for v in val:
            if isinstance(v, str):
                out.append(v)
            elif isinstance(v, dict):
                for k in ["raw_label", "label", "subcode", "code", "name"]:
                    if k in v:
                        out.append(str(v[k]))
                        break
        return out
    return []

def clean_alias(alias: str) -> str:
    if not isinstance(alias, str):
        return str(alias)
    if "(" in alias and ")" in alias:
        head = alias.split("(")[0].strip()
        if head:
            return head
    return alias.strip()

def main():
    toy = load_toyset(TOYSET_JSON)
    amap = read_alias_map(ALIAS_MAP)

    raw_col   = find_column(amap, RAW_COL_CANDIDATES)
    alias_col = find_column(amap, ALIAS_COL_CANDIDATES)

    raw_to_alias: Dict[str, str] = {}
    alias_to_subcodes: Dict[str, set] = defaultdict(set)
    for _, row in amap.iterrows():
        raw   = str(row[raw_col]).strip()
        alias = clean_alias(str(row[alias_col]))
        raw_to_alias[raw] = alias
        alias_to_subcodes[alias].add(raw)

    alias_counts   = Counter()
    subcode_counts = Counter()
    missing_raw = 0

    for rec in toy:
        alias_labels = [clean_alias(a) for a in extract_labels(rec, LABEL_FIELD)]
        if not alias_labels:
            continue
        raw_labels = extract_raw_labels(rec, RAW_FIELD) if RAW_FIELD else []

        if raw_labels:
            mapped_any = False
            for r in raw_labels:
                a = raw_to_alias.get(r)
                if a is None:
                    missing_raw += 1
                    for a2 in alias_labels:
                        alias_counts[a2] += 1
                    continue
                alias_counts[a] += 1
                subcode_counts[(r, a)] += 1
                mapped_any = True
            if not mapped_any:
                for a in alias_labels:
                    alias_counts[a] += 1
        else:
            for a in alias_labels:
                alias_counts[a] += 1

    rows = []
    for alias, total in sorted(alias_counts.items(), key=lambda x: (-x[1], x[0])):
        subcodes = sorted(list(alias_to_subcodes.get(alias, [])))
        rows.append({"alias": alias, "total": total, "n_subcodes": len(subcodes), "subcodes_csv": "; ".join(subcodes)})
    alias_df = pd.DataFrame(rows)
    alias_df.to_csv(f"{OUT_PREFIX}_alias_summary.csv", index=False)

    # sc_rows = [{"subcode": sc, "alias": al, "count": cnt} for (sc, al), cnt in subcode_counts.items()]
    # subcode_df = pd.DataFrame(sc_rows).sort_values(["alias", "count"], ascending=[True, False])
    # subcode_df.to_csv(f"{OUT_PREFIX}_subcode_breakdown.csv", index=False)
    sc_rows = [{"subcode": sc, "alias": al, "count": cnt} for (sc, al), cnt in subcode_counts.items()]
    subcode_df = pd.DataFrame(sc_rows)

    # If no raw subcodes were present (RAW_FIELD == "" or none matched), we won’t have a breakdown
    if subcode_df.empty:
        # still write an empty file with headers so downstream code won’t crash
        subcode_df = pd.DataFrame(columns=["subcode", "alias", "count"])
    else:
        subcode_df = subcode_df.sort_values(["alias", "count"], ascending=[True, False])

    subcode_df.to_csv(f"{OUT_PREFIX}_subcode_breakdown.csv", index=False)


    summary = {}
    for alias, grp in subcode_df.groupby("alias"):
        contrib = {r["subcode"]: int(r["count"]) for _, r in grp.iterrows()}
        summary[alias] = {
            "total": int(alias_counts.get(alias, 0)),
            "subcodes": sorted(list(alias_to_subcodes.get(alias, []))),
            "contributions": contrib
        }
    for alias in alias_counts:
        if alias not in summary:
            summary[alias] = {
                "total": int(alias_counts[alias]),
                "subcodes": sorted(list(alias_to_subcodes.get(alias, []))),
                "contributions": {}
            }

    with open(f"{OUT_PREFIX}_alias_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    plt.figure(figsize=(10, 5))
    xs = list(alias_counts.keys())
    ys = [alias_counts[a] for a in xs]
    plt.bar(xs, ys)
    plt.xticks(rotation=30, ha="right")
    plt.title("Interactional alias totals")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(f"{OUT_PREFIX}_alias_totals.png", dpi=200)

    print(f"Saved:\n- {OUT_PREFIX}_alias_summary.csv\n- {OUT_PREFIX}_subcode_breakdown.csv\n- {OUT_PREFIX}_alias_summary.json\n- {OUT_PREFIX}_alias_totals.png")
    if missing_raw:
        print(f"Note: {missing_raw} raw_label entries did not match the alias map and were counted at alias level only.")

if __name__ == "__main__":
    main()
