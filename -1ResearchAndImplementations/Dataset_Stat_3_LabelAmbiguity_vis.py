# Dataset_Stat_3_LabelAmbiguity_Figures.py
# Build figures for the Label Ambiguity subsection from the CSV outputs.

import os, csv, textwrap
from collections import Counter
import matplotlib.pyplot as plt

IN_DIR = "Data_for_Evidences/ambiguity_stats"
OUT_DIR = "Data_for_Evidences/ambiguity_stats/Images"
os.makedirs(OUT_DIR, exist_ok=True)

# ------------- Helper for wrapped text -------------
def wrap_txt(s, width=70):
    return "\n".join(textwrap.wrap(s, width=width, replace_whitespace=False, drop_whitespace=False))

# ------------- Figure 1: Nearest neighbor conflicts panel -------------
# Uses the first N rows from nn_conflict_samples.csv to create a 3 or 6 cell panel
NN_CSV = os.path.join(IN_DIR, "nn_conflict_samples.csv")
N_EXAMPLES = 3   # change to 6 if you want a 2x3 grid

examples = []
with open(NN_CSV, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        examples.append(row)

if not examples:
    print("[WARN] No rows in nn_conflict_samples.csv. Skipping nn panel.")
else:
    ex = examples[:N_EXAMPLES]
    n_rows = 1 if N_EXAMPLES <= 3 else 2
    n_cols = N_EXAMPLES if N_EXAMPLES <= 3 else (N_EXAMPLES + 1) // 2

    fig = plt.figure(figsize=(14, 4*n_rows))
    for i, row in enumerate(ex, 1):
        ax = fig.add_subplot(n_rows, n_cols, i)
        a_txt = wrap_txt(row["anchor_text"], 80)
        n_txt = wrap_txt(row["neighbor_text"], 80)
        a_lab = row["anchor_label"]
        n_lab = row["neighbor_label"]
        sim   = float(row["similarity"]) if row.get("similarity") else None

        block = (f"Anchor label: {a_lab}\n"
                 f"{a_txt}\n\n"
                 f"Neighbor label: {n_lab}\n"
                 f"{n_txt}\n\n"
                 f"Cosine similarity: {sim:.2f}" if sim is not None else "")
        ax.text(0.01, 0.99, block, va="top", ha="left", fontsize=10, family="monospace")
        ax.set_axis_off()

    fig.suptitle("Nearest-neighbor clauses with conflicting labels", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out1 = os.path.join(OUT_DIR, "nn_conflict_examples.png")
    fig.savefig(out1, dpi=250)
    plt.close(fig)
    print(f"[OK] Saved: {out1}")

# ------------- Figure 2: Ambiguous label pairs ranked bar chart -------------
PAIR_CSV = os.path.join(IN_DIR, "ambiguous_label_pairs.csv")
pairs = []
with open(PAIR_CSV, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        # expected columns: label_A, label_B, count
        try:
            cnt = int(row.get("count", "0"))
        except ValueError:
            cnt = 0
        pairs.append((f"{row.get('label_A','')} ↔ {row.get('label_B','')}", cnt))

pairs = [p for p in pairs if p[1] > 0]
pairs.sort(key=lambda x: x[1], reverse=True)
TOP = 15
pairs_top = pairs[:TOP]

if not pairs_top:
    print("[WARN] No pairs to plot in ambiguous_label_pairs.csv. Skipping bar plot.")
else:
    labels = [wrap_txt(p[0], 45) for p in pairs_top]
    counts = [p[1] for p in pairs_top]
    plt.figure(figsize=(12, max(4, 0.5*len(labels))))
    plt.barh(range(len(labels)), counts)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Conflict count among top-k neighbors")
    plt.title("Top ambiguous label pairs")
    plt.tight_layout()
    out2 = os.path.join(OUT_DIR, "ambiguous_pairs_bar.png")
    plt.savefig(out2, dpi=250)
    plt.close()
    print(f"[OK] Saved: {out2}")
