# toyset_analysis.py
import json
from collections import Counter
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

GOAL = "toysets_out/goal_toyset_raw.json"
INTER = "toysets_out/interactional_toyset_raw.json"
OUT = Path("toyset_analysis_out"); OUT.mkdir(parents=True, exist_ok=True)

def extract_label_and_type(item):
    primary = None
    if isinstance(item.get("selected_labels"), list) and item["selected_labels"]:
        primary = item["selected_labels"][0]
    ltype = None
    if isinstance(item.get("labels"), list) and item["labels"]:
        lbl = item["labels"][0]
        if primary is None and isinstance(lbl.get("label"), list) and lbl["label"]:
            primary = lbl["label"][0]
        if isinstance(lbl.get("label_type"), list) and lbl["label_type"]:
            ltype = lbl["label_type"][0]
    if item.get("label_type") and ltype is None:
        ltype = item["label_type"]
    return primary or "UNKNOWN", ltype

def analyze(path, name):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    total = len(data)
    lc, tc, sc = Counter(), Counter(), Counter()
    rows = []
    for k, it in data.items():
        label, ltype = extract_label_and_type(it)
        split = it.get("split", "unknown")
        lc[label] += 1
        if ltype: tc[ltype] += 1
        sc[split] += 1
        rows.append({"id": k, "text": it.get("text",""), "span": it.get("span",""),
                     "label": label, "label_type": ltype, "split": split})
    df = pd.DataFrame(rows)
    def dist_df(counter):
        return pd.DataFrame([{"key": k, "count": v, "percent": v/total*100}
                             for k, v in counter.most_common()])
    label_df = dist_df(lc).rename(columns={"key":"label"})
    type_df = dist_df(tc).rename(columns={"key":"label_type"})
    split_df = dist_df(sc).rename(columns={"key":"split"})
    summary = {
        "dataset": name,
        "total_items": total,
        "unique_labels": int(len(lc)),
        "labels": label_df.to_dict("records"),
        "label_types": type_df.to_dict("records"),
        "splits": split_df.to_dict("records"),
        "generated_at": datetime.utcnow().isoformat()+"Z"
    }
    # save
    df.to_csv(OUT / f"{name}_rows.csv", index=False)
    label_df.to_csv(OUT / f"{name}_label_dist.csv", index=False)
    with open(OUT / f"{name}_label_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return df, label_df, split_df, summary

def bar(df, x, y, title, outpath):
    import matplotlib.pyplot as plt
    plt.figure()
    s = df.sort_values(y, ascending=False)
    plt.bar(s[x].astype(str), s[y])
    plt.xticks(rotation=45, ha="right")
    plt.title(title); plt.xlabel(x); plt.ylabel(y)
    plt.tight_layout(); plt.savefig(outpath, dpi=200); plt.close()

g_df, g_lab, g_split, g_sum = analyze(GOAL, "goal_toyset_raw")
i_df, i_lab, i_split, i_sum = analyze(INTER, "interactional_toyset_raw")

with open(OUT / "combined_overview.json", "w", encoding="utf-8") as f:
    json.dump({"datasets":[g_sum, i_sum], "generated_at": datetime.utcnow().isoformat()+"Z"}, f, ensure_ascii=False, indent=2)

bar(g_lab, "label", "count", "Goal toyset: label counts", OUT / "goal_label_counts.png")
bar(i_lab, "label", "count", "Interactional toyset: label counts", OUT / "interactional_label_counts.png")
bar(g_split, "split", "count", "Goal toyset: split counts", OUT / "goal_split_counts.png")
bar(i_split, "split", "count", "Interactional toyset: split counts", OUT / "interactional_split_counts.png")

# side-by-side comparison figure
common = sorted(set(g_lab["label"]).union(set(i_lab["label"])))
g_map = dict(zip(g_lab["label"], g_lab["count"]))
i_map = dict(zip(i_lab["label"], i_lab["count"]))
import matplotlib.pyplot as plt
import numpy as np
vals = [(lab, int(g_map.get(lab,0)), int(i_map.get(lab,0))) for lab in common]
labs = [v[0] for v in vals]; gvals = [v[1] for v in vals]; ivals = [v[2] for v in vals]
x = np.arange(len(labs)); w = 0.4
plt.figure()
plt.bar(x - w/2, gvals, width=w, label="Goal")
plt.bar(x + w/2, ivals, width=w, label="Interactional")
plt.xticks(x, labs, rotation=45, ha="right")
plt.title("Label counts: Goal vs Interactional"); plt.xlabel("label"); plt.ylabel("count"); plt.legend()
plt.tight_layout(); plt.savefig(OUT / "goal_vs_interactional_label_counts.png", dpi=200); plt.close()

print("Done. Outputs in:", OUT)
