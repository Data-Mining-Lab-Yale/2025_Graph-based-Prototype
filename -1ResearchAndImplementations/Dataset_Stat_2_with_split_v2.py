# Dataset_Stat_2_TypeFlat.py
# Flat type-based analysis: Interactional vs Goal-Oriented
# Inputs:
#   - processed_messages_with_annotations.json
#   - annotation_code_mapping_detailed_corrected.json
#   - split_intents_by_type.json
#
# Outputs (in type_flat_stats/):
#   - interactional_label_counts.csv / .png
#   - goaloriented_label_counts.csv / .png
#   - interactional_metrics.json / .csv
#   - goaloriented_metrics.json / .csv
#   - type_comparison_metrics.csv  (summary table)

import json, os, math, csv
from collections import Counter
import matplotlib.pyplot as plt

# ========= Inputs (edit if needed) =========
ANNOTATION_FILE = "EPPC_output_json/CleanedData/processed_messages_with_annotations.json"
MAPPING_FILE    = "EPPC_output_json/Labels/annotation_code_mapping_detailed_corrected.json"
SPLIT_FILE      = "EPPC_output_json/Labels/split_intents_by_type.json"
OUTDIR          = "Data_for_Evidences/type_flat_stats"

# Plot controls
TOP_N = None          # set to an int for top-N bars; None to plot all
SHOW_PERCENT = False  # True to plot percentages
DPI = 200

os.makedirs(OUTDIR, exist_ok=True)

# ========= Load =========
with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    messages = json.load(f)
with open(MAPPING_FILE, "r", encoding="utf-8") as f:
    mapping = json.load(f)
with open(SPLIT_FILE, "r", encoding="utf-8") as f:
    split = json.load(f)

# ========= Helpers =========
def lookup_token(token: str):
    """Map a raw token to {'level': 'code'|'subcode'|..., 'matched_codebook_label': str}."""
    if token in mapping: return mapping[token]
    t_low, t_up = token.lower(), token.upper()
    if t_low in mapping: return mapping[t_low]
    if t_up in mapping:  return mapping[t_up]
    return None

def compute_imbalance_metrics(counter: Counter):
    """Return metrics for a flat label set."""
    k = len(counter)
    total = sum(counter.values())
    if k == 0 or total == 0:
        return {
            "num_classes": k, "total": total,
            "IR_max": None, "CV": None,
            "entropy": None, "entropy_normalized": None,
            "gini": None, "effective_classes": None,
            "majority_class_count": None, "minority_class_count": None,
        }
    counts = list(counter.values())
    majority, minority = max(counts), min(counts)
    IR_max = majority / minority if minority > 0 else math.inf
    mean = total / k
    var  = sum((c - mean)**2 for c in counts) / k
    std  = math.sqrt(var)
    CV   = std / mean if mean > 0 else None
    ps = [c / total for c in counts]
    eps = 1e-12
    H   = -sum(p * math.log(p + eps) for p in ps)
    Hn  = H / math.log(k) if k > 1 else 0.0
    G   = 1.0 - sum(p * p for p in ps)
    Ke  = 1.0 / sum(p * p for p in ps)
    return {
        "num_classes": k, "total": total,
        "IR_max": IR_max, "CV": CV,
        "entropy": H, "entropy_normalized": Hn,
        "gini": G, "effective_classes": Ke,
        "majority_class_count": majority, "minority_class_count": minority,
    }

def plot_bar(counter: Counter, title: str, filename: str, show_percent=False, top_n=None):
    if not counter:
        print(f"[WARN] No data to plot for: {title}")
        return
    items = counter.most_common(top_n) if top_n else counter.most_common()
    labels, counts = zip(*items)
    if show_percent:
        total = sum(counts)
        values = [c / total * 100 for c in counts]
        ylabel = "Percentage (%)"
    else:
        values = counts
        ylabel = "Count"
    plt.figure(figsize=(max(8, 0.45 * len(labels)), 6))
    plt.bar(labels, values)
    plt.xticks(rotation=60, ha="right")
    plt.ylabel(ylabel); plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=DPI); plt.close()
    print(f"[OK] Saved: {filename}")

def save_counts(counter: Counter, filename: str):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "count"])
        for lab, cnt in counter.most_common():
            w.writerow([lab, cnt])
    print(f"[OK] Saved: {filename}")

def save_metrics(metrics: dict, filename_json: str, filename_csv: str):
    with open(filename_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(filename_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["metric", "value"])
        for k, v in metrics.items():
            w.writerow([k, v])
    print(f"[OK] Saved: {filename_json}, {filename_csv}")

# ========= Build flat label sets from split file =========
# We will treat both Interactional and Goal-Oriented as flat sets of label strings.
# The split items may include entries with "level": "code" or "subcode".
# For flat counting, we accept either. The label string in 'label' is the unit we count.

def build_flat_label_set(split_block):
    """Return a set of labels (strings) to count for this type."""
    flat = set()
    for item in split_block:
        lab = item.get("label")
        if lab: flat.add(lab)
    return flat

INTER_LABS = build_flat_label_set(split.get("Interactional", []))
GOAL_LABS  = build_flat_label_set(split.get("Goal-Oriented", []))

print(f"[INFO] Interactional labels: {len(INTER_LABS)}")
print(f"[INFO] Goal-Oriented labels: {len(GOAL_LABS)}")

# ========= Core counting logic =========
def process_type_flat(type_name: str, allowed_labels: set):
    """
    Count examples per label inside the allowed set.
    Mapping rule:
      - For each annotation, map tokens to code_label and subcode_label if present.
      - Prefer counting subcode_label when it exists and is in the allowed set.
      - If there is no subcode_label (or not in set) but code_label is in the set, count the code_label.
      - If both map into the set, count the subcode (more specific) and ignore the code to avoid double count.
    """
    per_label = Counter()
    total_items = 0
    mapped_items = 0

    for msg in messages:
        for ann in msg.get("annotations", []):
            total_items += 1
            raw_tokens = ann.get("code", [])
            if not isinstance(raw_tokens, list):
                continue

            code_label, subcode_label = None, None
            for tok in raw_tokens:
                info = lookup_token(tok)
                if not info:
                    continue
                lvl = info.get("level")
                matched = info.get("matched_codebook_label")
                if lvl == "code" and code_label is None:
                    code_label = matched
                elif lvl == "subcode" and subcode_label is None:
                    subcode_label = matched

            # Decide which label to count in this flat type
            chosen = None
            if subcode_label and subcode_label in allowed_labels:
                chosen = subcode_label
            elif code_label and code_label in allowed_labels:
                chosen = code_label

            if chosen is not None:
                mapped_items += 1
                per_label[chosen] += 1

    print(f"\n===== {type_name} (flat) =====")
    print(f"[INFO] Total annotation items: {total_items}")
    print(f"[INFO] Items counted in this type: {mapped_items}")

    # Metrics and saves
    metrics = compute_imbalance_metrics(per_label)

    counts_csv = os.path.join(OUTDIR, f"{type_name.lower()}_label_counts.csv")
    save_counts(per_label, counts_csv)

    metrics_json = os.path.join(OUTDIR, f"{type_name.lower()}_metrics.json")
    metrics_csv  = os.path.join(OUTDIR, f"{type_name.lower()}_metrics.csv")
    save_metrics(metrics, metrics_json, metrics_csv)

    plot_bar(
        per_label,
        f"{type_name}: Label distribution (flat)",
        os.path.join(OUTDIR, f"{type_name.lower()}_label_counts.png"),
        show_percent=SHOW_PERCENT,
        top_n=TOP_N
    )

    # Pretty print
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    return per_label, metrics

inter_counts, inter_metrics = process_type_flat("Interactional", INTER_LABS)
goal_counts,  goal_metrics  = process_type_flat("GoalOriented",  GOAL_LABS)

# ========= Comparison table =========
with open(os.path.join(OUTDIR, "type_comparison_metrics.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["type", "num_classes", "total", "IR_max", "CV", "entropy_normalized", "gini", "effective_classes"])
    for tname, m in [("Interactional", inter_metrics), ("GoalOriented", goal_metrics)]:
        w.writerow([
            tname,
            m.get("num_classes"),
            m.get("total"),
            f"{m.get('IR_max'):.6f}" if isinstance(m.get("IR_max"), float) else m.get("IR_max"),
            f"{m.get('CV'):.6f}" if isinstance(m.get("CV"), float) else m.get("CV"),
            f"{m.get('entropy_normalized'):.6f}" if isinstance(m.get("entropy_normalized"), float) else m.get("entropy_normalized"),
            f"{m.get('gini'):.6f}" if isinstance(m.get("gini"), float) else m.get("gini"),
            f"{m.get('effective_classes'):.6f}" if isinstance(m.get("effective_classes"), float) else m.get("effective_classes"),
        ])
print(f"\n[DONE] Outputs in: {OUTDIR}")
