"""
WOVEN Step 6 — Result Stats (JSON-first, simple CASE_STEM switch)
-----------------------------------------------------------------
What it does:
1) Loads annotated split data for a chosen CASE_STEM (no in-data string checks).
2) Ensures flags exist:
   - patient-to-provider, provider-to-patient, provider-to-provider,
     is_telephone_note, is_content, is_multiple
3) Derives:
   - role_sum, multi_from_roles
   - label_combo using FULL LABEL NAMES (or joined with '+')
4) Aggregates:
   - flag counts, label_combo counts, multi-label rate
   - decision tree counts for path: is_telephone_note -> is_content -> label_combo
   - by Case Type (if column exists)
   - mismatch report: is_multiple vs multi_from_roles
5) Saves:
   - analysis_summary.json
   - combo_counts.csv
   - decision_tree_counts.csv/.json
   - by_case_type_combo.csv (if available)
   - combo_counts_bar.png (with counts)
   - decision_tree_counts.png (schematic)
"""

import os
import io
import json
import pandas as pd
import matplotlib.pyplot as plt

# ========= PICK WHICH SET =========
CASE_STEM = "Statement_Billing_or_Insurance_Question"   # e.g., "Clinical_Question", "Medication", "Statement_Billing_or_Insurance_Question"

# ========= INPUT / OUTPUT PATHS (no normalization, no in-data filters) =========
BASE_DIR = "Data/WOVEN/split_by_case_type/checked_results"
CANDIDATES = [
    os.path.join(BASE_DIR, f"{CASE_STEM}_annotated_output.jsonl"),
    os.path.join(BASE_DIR, f"{CASE_STEM}_annotated_output.json"),
    os.path.join(BASE_DIR, f"{CASE_STEM}_annotated_output.csv"),
]
INPUT_PATH = next((p for p in CANDIDATES if os.path.exists(p)), CANDIDATES[0])

OUT_DIR = os.path.join("Data/WOVEN/analysis_outputs", CASE_STEM)
os.makedirs(OUT_DIR, exist_ok=True)

# ========= FLAGS =========
FLAG_COLS = [
    "patient-to-provider",
    "provider-to-patient",
    "provider-to-provider",
    "is_telephone_note",
    "is_content",
    "is_multiple",
]
ROLE_COLS = ["patient-to-provider", "provider-to-patient", "provider-to-provider"]

# ========= LOADER =========
def read_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input not found: {path}")
    if ext == ".jsonl":
        with open(path, "rb") as f:
            return pd.read_json(io.BytesIO(f.read()), lines=True)
    elif ext == ".json":
        # Try JSON Lines first, then array
        try:
            with open(path, "rb") as f:
                return pd.read_json(io.BytesIO(f.read()), lines=True)
        except ValueError:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return pd.json_normalize(data, sep=".")
            raise ValueError("JSON must be JSONL or an array of objects.")
    elif ext == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError("Expected .jsonl/.json/.csv")

# ========= UTIL =========
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

def derive_combo_full(row) -> str:
    roles_on = [c for c in ROLE_COLS if int(row.get(c, 0)) == 1]
    if len(roles_on) == 0:
        return "none"
    if len(roles_on) == 1:
        return roles_on[0]
    return "+".join(sorted(roles_on))

def leaf_summary(d: pd.DataFrame) -> str:
    if d is None or len(d) == 0:
        return "—"
    vc = d["label_combo"].value_counts()
    # show each label and count on its own line for readability
    lines = [f"{k}: {int(vc[k])}" for k in sorted(vc.index)]
    return "\n".join(lines)

# ========= MAIN =========
print(f"Loading: {INPUT_PATH}")
df = read_any(INPUT_PATH)
if df is None or len(df) == 0:
    raise ValueError("No rows loaded from input file.")

df = ensure_flags(df)

# Derived columns
df["role_sum"] = df[ROLE_COLS].sum(axis=1)
df["multi_from_roles"] = (df["role_sum"] > 1).astype(int)
df["label_combo"] = df.apply(derive_combo_full, axis=1)

# Basic stats
N = len(df)
flag_counts = {c: int(df[c].sum()) for c in FLAG_COLS}
combo_counts = df["label_combo"].value_counts().sort_index()
multi_rate_roles = float(df["multi_from_roles"].mean()) if N else 0.0

# By Case Type if present
if "Case Type" in df.columns:
    by_case_type_combo = df.pivot_table(index="Case Type", columns="label_combo",
                                        values="role_sum", aggfunc="count", fill_value=0).reset_index()
else:
    by_case_type_combo = pd.DataFrame()

# Mismatch report: is_multiple vs role-derived multiple
df["is_multiple_matches_roles"] = (df["is_multiple"] == df["multi_from_roles"]).astype(int)
mismatches = df[df["is_multiple"] != df["multi_from_roles"]].copy()
mismatch_csv = os.path.join(OUT_DIR, f"{CASE_STEM}_is_multiple_mismatch.csv")
mismatches.to_csv(mismatch_csv, index=False)

# Decision tree path counts (telephone -> content -> label_combo)
paths = []
unique_leaves = sorted(df["label_combo"].unique())
for tel in [0, 1]:
    d1 = df[df["is_telephone_note"] == tel]
    for content in [0, 1]:
        d2 = d1[d1["is_content"] == content]
        vc = d2["label_combo"].value_counts()
        for leaf in unique_leaves:
            cnt = int(vc.get(leaf, 0))
            paths.append({
                "is_telephone_note": int(tel),
                "is_content": int(content),
                "label_combo": leaf,
                "count": cnt,
                "percent_of_all": (cnt / N * 100.0) if N else 0.0
            })
decision_df = pd.DataFrame(paths).sort_values(
    by=["is_telephone_note", "is_content", "label_combo"]
).reset_index(drop=True)

# Save summaries
summary = {
    "case_stem": CASE_STEM,
    "input_file": INPUT_PATH,
    "total_rows": N,
    "flag_counts": flag_counts,
    "label_combo_counts": combo_counts.to_dict(),
    "multi_label_rate_from_roles": multi_rate_roles,
    "is_multiple_agreement_rate": float(df["is_multiple_matches_roles"].mean()) if N else 0.0,
    "mismatch_rows": int(len(mismatches)),
}
with open(os.path.join(OUT_DIR, f"{CASE_STEM}_analysis_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

combo_counts.to_csv(os.path.join(OUT_DIR, f"{CASE_STEM}_combo_counts.csv"), header=["count"])
decision_df.to_csv(os.path.join(OUT_DIR, f"{CASE_STEM}_decision_tree_counts.csv"), index=False)
decision_df.to_json(os.path.join(OUT_DIR, f"{CASE_STEM}_decision_tree_counts.json"), orient="records", indent=2)

if not by_case_type_combo.empty:
    by_case_type_combo.to_csv(os.path.join(OUT_DIR, f"{CASE_STEM}_by_case_type_combo.csv"), index=False)

# ========= PLOTS =========
# Bar chart with labels
plt.figure(figsize=(10, 6))
counts_sorted = combo_counts.sort_values(ascending=False)
ax = counts_sorted.plot(kind="bar")
plt.title(f"Label Combo Counts — {CASE_STEM}")
plt.xlabel("label_combo")
plt.ylabel("count")
for i, v in enumerate(counts_sorted):
    ax.text(i, v + 0.5, str(v), ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"{CASE_STEM}_combo_counts_bar.png"), dpi=160)
plt.close()

# Simple decision tree schematic
def draw_tree(df_local: pd.DataFrame, out_path: str):
    root = df_local
    tel0 = root[root["is_telephone_note"] == 0]
    tel1 = root[root["is_telephone_note"] == 1]
    tel0_c0 = tel0[tel0["is_content"] == 0]
    tel0_c1 = tel0[tel0["is_content"] == 1]
    tel1_c0 = tel1[tel1["is_content"] == 0]
    tel1_c1 = tel1[tel1["is_content"] == 1]

    plt.figure(figsize=(11, 7))
    ax = plt.gca()
    ax.axis("off")

    # positions
    rx, ry = 0.5, 0.95
    t0x, t0y = 0.25, 0.75
    t1x, t1y = 0.75, 0.75
    t0c0x, t0c0y = 0.13, 0.50
    t0c1x, t0c1y = 0.37, 0.50
    t1c0x, t1c0y = 0.63, 0.50
    t1c1x, t1c1y = 0.87, 0.50

    def box(x, y, text):
        ax.text(x, y, text, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.4", fc="white"), fontsize=9)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-"))

    box(rx, ry, f"ALL\nN={len(root)}")
    arrow(rx, ry-0.02, t0x, t0y+0.03)
    arrow(rx, ry-0.02, t1x, t1y+0.03)
    box(t0x, t0y, f"tel=0\nN={len(tel0)}")
    box(t1x, t1y, f"tel=1\nN={len(tel1)}")

    arrow(t0x, t0y-0.02, t0c0x, t0c0y+0.03)
    arrow(t0x, t0y-0.02, t0c1x, t0c1y+0.03)
    arrow(t1x, t1y-0.02, t1c0x, t1c0y+0.03)
    arrow(t1x, t1y-0.02, t1c1x, t1c1y+0.03)

    box(t0c0x, t0c0y, f"tel=0, content=0\nN={len(tel0_c0)}\n{leaf_summary(tel0_c0)}")
    box(t0c1x, t0c1y, f"tel=0, content=1\nN={len(tel0_c1)}\n{leaf_summary(tel0_c1)}")
    box(t1c0x, t1c0y, f"tel=1, content=0\nN={len(tel1_c0)}\n{leaf_summary(tel1_c0)}")
    box(t1c1x, t1c1y, f"tel=1, content=1\nN={len(tel1_c1)}\n{leaf_summary(tel1_c1)}")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

draw_tree(df, os.path.join(OUT_DIR, f"{CASE_STEM}_decision_tree_counts.png"))

# Final report
print("=== Stats complete ===")
print(f"- Summary JSON: {os.path.join(OUT_DIR, CASE_STEM + '_analysis_summary.json')}")
print(f"- Decision tree (CSV/JSON): {os.path.join(OUT_DIR, CASE_STEM + '_decision_tree_counts.csv')}")
print(f"- Combo counts CSV: {os.path.join(OUT_DIR, CASE_STEM + '_combo_counts.csv')}")
if not by_case_type_combo.empty:
    print(f"- By case type CSV: {os.path.join(OUT_DIR, CASE_STEM + '_by_case_type_combo.csv')}")
print(f"- Charts: {os.path.join(OUT_DIR, CASE_STEM + '_combo_counts_bar.png')}, {os.path.join(OUT_DIR, CASE_STEM + '_decision_tree_counts.png')}")
