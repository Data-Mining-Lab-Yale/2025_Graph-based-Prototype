# Dataset_Stat_3_AmbiguityPairs_VerticalBar.py
import os, csv, textwrap
import matplotlib.pyplot as plt

IN_CSV = os.path.join("Data_for_Evidences/ambiguity_stats", "ambiguous_label_pairs.csv")
OUT_IMG = os.path.join("Data_for_Evidences/ambiguity_stats/Images", "ambiguous_pairs_bar_vertical.png")
TOP = 15  # show top-N pairs

os.makedirs("Images", exist_ok=True)

pairs = []
with open(IN_CSV, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        try:
            cnt = int(row.get("count", "0"))
        except ValueError:
            cnt = 0
        a = row.get("label_A", "")
        b = row.get("label_B", "")
        pairs.append((f"{a} ↔ {b}", cnt))

pairs = [p for p in pairs if p[1] > 0]
pairs.sort(key=lambda x: x[1], reverse=True)
pairs = pairs[:TOP]

labels = [textwrap.fill(p[0], width=35) for p in pairs]
counts = [p[1] for p in pairs]

plt.figure(figsize=(max(10, 0.6*len(labels)), 6))
plt.bar(labels, counts)  # vertical bars
plt.xticks(rotation=60, ha="right")
plt.ylabel("Conflict count among top-k neighbors")
plt.title("Top ambiguous label pairs")
plt.tight_layout()
plt.savefig(OUT_IMG, dpi=250)
plt.close()
print(f"[OK] Saved: {OUT_IMG}")
