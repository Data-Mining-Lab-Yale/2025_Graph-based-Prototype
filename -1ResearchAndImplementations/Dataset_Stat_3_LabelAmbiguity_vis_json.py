# Dataset_Stat_3_NN_Conflicts_ToJSON.py
import os, csv, json

IN_CSV  = os.path.join("Data_for_Evidences/ambiguity_stats", "nn_conflict_samples.csv")
OUT_JSON_FLAT = os.path.join("Data_for_Evidences/ambiguity_stats", "nn_conflicts_examples.json")
OUT_JSON_BY_PAIR = os.path.join("Data_for_Evidences/ambiguity_stats", "nn_conflicts_by_pair.json")

# optional controls
MAX_PER_PAIR = 50   # cap stored examples per pair to keep files manageable
DEDUP = True        # remove exact duplicate (anchor, neighbor, labels, sim) rows

flat = []
by_pair = {}

seen = set()

with open(IN_CSV, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        ex = {
            "anchor_text": row.get("anchor_text", ""),
            "anchor_label": row.get("anchor_label", ""),
            "neighbor_text": row.get("neighbor_text", ""),
            "neighbor_label": row.get("neighbor_label", ""),
            "similarity": float(row["similarity"]) if row.get("similarity") else None,
        }
        if DEDUP:
            key = (ex["anchor_text"], ex["anchor_label"], ex["neighbor_text"], ex["neighbor_label"], ex["similarity"])
            if key in seen:
                continue
            seen.add(key)

        # flat list
        flat.append(ex)

        # grouped by pair
        a, b = sorted([ex["anchor_label"], ex["neighbor_label"]])
        pair_key = f"{a}__VS__{b}"
        by_pair.setdefault(pair_key, [])
        if len(by_pair[pair_key]) < MAX_PER_PAIR:
            by_pair[pair_key].append(ex)

with open(OUT_JSON_FLAT, "w", encoding="utf-8") as f:
    json.dump(flat, f, indent=2, ensure_ascii=False)
with open(OUT_JSON_BY_PAIR, "w", encoding="utf-8") as f:
    json.dump(by_pair, f, indent=2, ensure_ascii=False)

print(f"[OK] Saved: {OUT_JSON_FLAT}")
print(f"[OK] Saved: {OUT_JSON_BY_PAIR}")
