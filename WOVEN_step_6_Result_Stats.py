"""
WOVEN analysis + decision-tree counts (JSON-first, CSV allowed)
---------------------------------------------------------------
What it does:
1) Loads annotated data (JSON/JSONL or CSV).
2) Ensures 5 flags exist:
   - patient-to-provider, provider-to-patient, provider-to-provider, is_telephone_note, is_content
3) Derives:
   - label_sum, has_multiple_labels
   - label_combo (pt2pr / pr2pt / pr2pr / multi / none)
4) Aggregates counts:
   - global flag counts
   - label_combo counts
   - per Case Type summaries
   - decision-tree style path:
     is_telephone_note -> is_content -> label_combo
5) Saves:
   - analysis_summary.json (key stats)
   - decision_tree_counts.csv / .json
   - combo_counts.csv
   - by_case_type_combo.csv
6) Plots:
   - bar chart for label_combo counts
   - simple tree figure with counts

Charts use matplotlib (no seaborn, no custom colors).
"""

import os
import json
import math
import pandas as pd
import matplotlib.pyplot as plt


# ======= CASE TYPE FILTER (None to load all) =======
# Examples: None, "Clinical_Question", "Medication", "Other"
CASE_TYPE_FILTER = "Clinical_Question"

# ====== INPUT / OUTPUT ======
INPUT_PATH = f"Data/WOVEN/split_by_case_type/{CASE_TYPE_FILTER}_annotated_output.jsonl"   # set to your annotated file (.jsonl / .json / .csv)
OUT_DIR = f"Data/WOVEN/{CASE_TYPE_FILTER}_analysis_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ====== REQUIRED FLAGS ======
FLAG_COLS = [
    "patient-to-provider",
    "provider-to-patient",
    "provider-to-provider",
    "is_telephone_note",
    "is_content",
]

# ====== LOAD ======
ext = os.path.splitext(INPUT_PATH)[1].lower()
if ext in [".jsonl", ".json"]:
    try:
        df = pd.read_json(INPUT_PATH, lines=True)
    except ValueError:
        # fallback: array-of-objects JSON
        df = pd.read_json(INPUT_PATH)
elif ext == ".csv":
    df = pd.read_csv(INPUT_PATH)
else:
    raise ValueError("INPUT_PATH must be .jsonl, .json, or .csv")

# ====== ENSURE FLAGS ======
for c in FLAG_COLS:
    if c not in df.columns:
        # sensible defaults for missing columns
        default = 1 if c in ["is_telephone_note", "is_content"] else (1 if c == "provider-to-provider" else 0)
        df[c] = default
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

# ====== DERIVED COLUMNS ======
role_cols = ["patient-to-provider", "provider-to-patient", "provider-to-provider"]
df["label_sum"] = df[role_cols].sum(axis=1)
df["has_multiple_labels"] = (df["label_sum"] > 1).astype(int)

def _combo(row):
    roles = [rc for rc in role_cols if row.get(rc, 0) == 1]
    if len(roles) == 0:
        return "none"
    if len(roles) == 1:
        # shorten names for readability
        short = {
            "patient-to-provider": "pt2pr",
            "provider-to-patient": "pr2pt",
            "provider-to-provider": "pr2pr",
        }
        return short[roles[0]]
    # multi-label
    short_map = {
        "patient-to-provider": "pt2pr",
        "provider-to-patient": "pr2pt",
        "provider-to-provider": "pr2pr",
    }
    return "+".join(sorted(short_map[r] for r in roles))
df["label_combo"] = df.apply(_combo, axis=1)

# ====== BASIC STATS ======
N = len(df)
flag_counts = {c: int(df[c].sum()) for c in FLAG_COLS}
combo_counts = df["label_combo"].value_counts().sort_index()
multi_rate = float(df["has_multiple_labels"].mean()) if N else 0.0

# By case type (if exists)
if "Case Type" in df.columns:
    by_case_type_combo = df.pivot_table(index="Case Type", columns="label_combo",
                                        values="label_sum", aggfunc="count", fill_value=0).reset_index()
else:
    by_case_type_combo = pd.DataFrame()

# ====== DECISION TREE COUNT TABLE ======
# Path: is_telephone_note (0/1) -> is_content (0/1) -> label_combo
paths = []
for tel in [0, 1]:
    d1 = df[df["is_telephone_note"] == tel]
    for content in [0, 1]:
        d2 = d1[d1["is_content"] == content]
        # leaf splits
        leaf_counts = d2["label_combo"].value_counts()
        # We want consistent order
        for leaf in sorted(df["label_combo"].unique()):
            cnt = int(leaf_counts.get(leaf, 0))
            paths.append({
                "is_telephone_note": tel,
                "is_content": content,
                "label_combo": leaf,
                "count": cnt,
                "percent_of_all": (cnt / N * 100.0) if N else 0.0
            })

decision_df = pd.DataFrame(paths).sort_values(by=["is_telephone_note", "is_content", "label_combo"]).reset_index(drop=True)

# ====== SAVE SUMMARIES ======
summary = {
    "total_rows": N,
    "flag_counts": flag_counts,
    "label_combo_counts": combo_counts.to_dict(),
    "multi_label_rate": multi_rate,
}
with open(os.path.join(OUT_DIR, "analysis_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

combo_counts.to_csv(os.path.join(OUT_DIR, "combo_counts.csv"), header=["count"])
decision_df.to_csv(os.path.join(OUT_DIR, "decision_tree_counts.csv"), index=False)
decision_df.to_json(os.path.join(OUT_DIR, "decision_tree_counts.json"), orient="records", indent=2)

if not by_case_type_combo.empty:
    by_case_type_combo.to_csv(os.path.join(OUT_DIR, "by_case_type_combo.csv"), index=False)

# ====== PLOTS ======
# 1) Bar chart for label_combo
plt.figure(figsize=(8, 5))
combo_counts.sort_values(ascending=False).plot(kind="bar")
plt.title("Label Combo Counts")
plt.xlabel("label_combo")
plt.ylabel("count")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "combo_counts_bar.png"), dpi=160)
plt.close()

# 2) Simple decision-tree diagram with counts
# We draw a small static tree: Root -> tel=0/1 -> content=0/1 -> leaf distro text
# This is not sklearn; it's a visual count tree.
def _leaf_summary(d):
    # return a compact "pt2pr:x  pr2pt:y  pr2pr:z  multi:w  none:k"
    order = ["pt2pr", "pr2pt", "pr2pr", "multi", "none"]
    vc = d["label_combo"].value_counts()
    parts = []
    for k in order:
        if k in vc:
            parts.append(f"{k}:{int(vc[k])}")
    return "  ".join(parts) if parts else "—"

# Compute subsets
root = df
tel0 = root[root["is_telephone_note"] == 0]
tel1 = root[root["is_telephone_note"] == 1]
tel0_c0 = tel0[tel0["is_content"] == 0]
tel0_c1 = tel0[tel0["is_content"] == 1]
tel1_c0 = tel1[tel1["is_content"] == 0]
tel1_c1 = tel1[tel1["is_content"] == 1]

plt.figure(figsize=(10, 6))
ax = plt.gca()
ax.axis("off")

# Node positions
# Root
rx, ry = 0.5, 0.95
# Level 1 (telephone)
t0x, t0y = 0.25, 0.75
t1x, t1y = 0.75, 0.75
# Level 2 (content)
t0c0x, t0c0y = 0.13, 0.50
t0c1x, t0c1y = 0.37, 0.50
t1c0x, t1c0y = 0.63, 0.50
t1c1x, t1c1y = 0.87, 0.50
# Leaf text boxes
def box(x, y, text):
    ax.text(x, y, text, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.4", fc="white"), fontsize=9)

# Lines
def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-"))

# Draw nodes and edges
box(rx, ry, f"ALL\nN={len(root)}")
arrow(rx, ry-0.02, t0x, t0y+0.03)
arrow(rx, ry-0.02, t1x, t1y+0.03)
box(t0x, t0y, f"tel=0\nN={len(tel0)}")
box(t1x, t1y, f"tel=1\nN={len(tel1)}")

arrow(t0x, t0y-0.02, t0c0x, t0c0y+0.03)
arrow(t0x, t0y-0.02, t0c1x, t0c1y+0.03)
arrow(t1x, t1y-0.02, t1c0x, t1c0y+0.03)
arrow(t1x, t1y-0.02, t1c1x, t1c1y+0.03)

# box(t0c0x, t0c0y, f"tel=0, content=0\nN={len(tel0_c0)}\n{_leaf_summary(tel0_c0)}")
# box(t0c1x, t0c1y, f"tel=0, content=1\nN={len(tel0_c1)}\n{_leaf_summary(tel0_c1)}")
# box(t1c0x, t1c0y, f"tel=1, content=0\nN={len(tel1_c0)}\n{_leaf_summary(tel1_c0)}")
# box(t1c1x, t1c1y, f"tel=1, content=1\nN={len(t1_c1)}\n{_leaf_summary(t1c1)}")
box(t0c0x, t0c0y, f"tel=0, content=0\nN={len(tel0_c0)}\n{_leaf_summary(tel0_c0)}")
box(t0c1x, t0c1y, f"tel=0, content=1\nN={len(tel0_c1)}\n{_leaf_summary(tel0_c1)}")
box(t1c0x, t1c0y, f"tel=1, content=0\nN={len(tel1_c0)}\n{_leaf_summary(tel1_c0)}")
box(t1c1x, t1c1y, f"tel=1, content=1\nN={len(tel1_c1)}\n{_leaf_summary(tel1_c1)}")



plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "decision_tree_counts.png"), dpi=160)
plt.close()

print("Done.")
print(f"- Summary JSON: {os.path.join(OUT_DIR, 'analysis_summary.json')}")
print(f"- Decision tree (CSV/JSON): {os.path.join(OUT_DIR, 'decision_tree_counts.csv')}")
print(f"- Combo counts CSV: {os.path.join(OUT_DIR, 'combo_counts.csv')}")
if not by_case_type_combo.empty:
    print(f"- By case type CSV: {os.path.join(OUT_DIR, 'by_case_type_combo.csv')}")
print(f"- Charts: {os.path.join(OUT_DIR, 'combo_counts_bar.png')}, {os.path.join(OUT_DIR, 'decision_tree_counts.png')}")
