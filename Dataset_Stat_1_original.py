# Dataset_Stat_1_Distributions.py
import json
import os
import math
import csv
from collections import Counter
import matplotlib.pyplot as plt

# ==== Input files (update paths if needed) ====
ANNOTATION_FILE = "EPPC_output_json/CleanedData/processed_messages_with_annotations.json"
MAPPING_FILE = "EPPC_output_json/Labels/annotation_code_mapping_detailed_corrected.json"
NODE_FILE = "EPPC_output_json/Labels/node_names_by_type_with_index.json"

# ========= Config =========
TOP_N = None          # set an int to only show top-N bars, or None for all
SHOW_PERCENT = False  # True to plot percentages instead of raw counts
DPI = 200

# ========= Load files =========
with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    messages = json.load(f)

with open(MAPPING_FILE, "r", encoding="utf-8") as f:
    mapping = json.load(f)

node_index = None
if os.path.exists(NODE_FILE):
    with open(NODE_FILE, "r", encoding="utf-8") as f:
        node_index = json.load(f)

# ========= Helpers =========
def lookup_token(token: str):
    """Find token in mapping with small robustness on casing."""
    if token in mapping:
        return mapping[token]
    t_low = token.lower()
    t_up = token.upper()
    if t_low in mapping:
        return mapping[t_low]
    if t_up in mapping:
        return mapping[t_up]
    return None

def plot_bar(counter: Counter, title: str, filename: str, show_percent=False, top_n=None):
    """Vertical bar chart sorted by count desc. Safe on empty counters."""
    if not counter:
        print(f"[WARN] No data to plot for: {title}")
        return
    items = counter.most_common(top_n) if top_n else counter.most_common()
    labels, counts = zip(*items)

    if show_percent:
        total = sum(counts)
        values = [c / total * 100.0 for c in counts]
        ylabel = "Percentage (%)"
    else:
        values = counts
        ylabel = "Count"

    plt.figure(figsize=(max(8, 0.4 * len(labels)), 6))
    plt.bar(labels, values)
    plt.xticks(rotation=60, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=DPI)
    plt.close()
    print(f"[OK] Saved: {filename}")

def save_csv(counter: Counter, filename: str):
    """Save counts to CSV with columns: label,count. Safe on empty counters."""
    with open(filename, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "count"])
        for label, cnt in counter.most_common():
            w.writerow([label, cnt])
    print(f"[OK] Saved: {filename}")

def save_metrics(metrics: dict, filename_json: str, filename_csv: str):
    """Save metrics to JSON and CSV."""
    with open(filename_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(filename_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in metrics.items():
            w.writerow([k, v])
    print(f"[OK] Saved: {filename_json}, {filename_csv}")

def compute_imbalance_metrics(counter: Counter):
    """Return IR_max, CV, entropy, entropy_norm, gini, effective_classes, k"""
    k = len(counter)
    total = sum(counter.values())
    if k == 0 or total == 0:
        return {
            "num_classes": k,
            "total": total,
            "IR_max": None,
            "CV": None,
            "entropy": None,
            "entropy_normalized": None,
            "gini": None,
            "effective_classes": None,
            "majority_class_count": None,
            "minority_class_count": None,
        }

    counts = list(counter.values())
    majority = max(counts)
    minority = min(counts)
    IR_max = majority / minority if minority > 0 else math.inf

    # CV = std / mean of counts
    mean = total / k
    var = sum((c - mean) ** 2 for c in counts) / k
    std = math.sqrt(var)
    CV = std / mean if mean > 0 else None

    # proportions
    ps = [c / total for c in counts]

    # Shannon entropy
    eps = 1e-12
    entropy = -sum(p * math.log(p + eps) for p in ps)
    entropy_norm = entropy / math.log(k) if k > 1 else 0.0

    # Gini = 1 - sum p_i^2
    gini = 1.0 - sum(p * p for p in ps)

    # Effective number of classes = 1 / sum p_i^2
    effective_classes = 1.0 / sum(p * p for p in ps)

    return {
        "num_classes": k,
        "total": total,
        "IR_max": IR_max,
        "CV": CV,
        "entropy": entropy,
        "entropy_normalized": entropy_norm,
        "gini": gini,
        "effective_classes": effective_classes,
        "majority_class_count": majority,
        "minority_class_count": minority,
    }

# ========= Count loops =========
code_counter = Counter()
subcode_counter = Counter()
combined_counter = Counter()

total_annotations = 0
mapped_annotations = 0

for msg in messages:
    for ann in msg.get("annotations", []):
        total_annotations += 1
        raw_tokens = ann.get("code", [])
        if not isinstance(raw_tokens, list):
            continue

        code_label = None
        subcode_label = None

        for tok in raw_tokens:
            info = lookup_token(tok)
            if not info:
                continue
            level = info.get("level")
            matched = info.get("matched_codebook_label")
            if level == "code" and code_label is None:
                code_label = matched
            elif level == "subcode" and subcode_label is None:
                subcode_label = matched

        if code_label or subcode_label:
            mapped_annotations += 1

        if code_label:
            code_counter[code_label] += 1
        if subcode_label:
            subcode_counter[subcode_label] += 1

        if code_label and subcode_label:
            combined_counter[f"{code_label}_{subcode_label}"] += 1
        elif code_label:
            combined_counter[code_label] += 1
        elif subcode_label:
            combined_counter[subcode_label] += 1

print(f"[INFO] Total annotation items: {total_annotations}")
print(f"[INFO] Items with at least one mapped token: {mapped_annotations}")

# ========= Metrics =========
metrics_code = compute_imbalance_metrics(code_counter)
metrics_subcode = compute_imbalance_metrics(subcode_counter)
metrics_combined = compute_imbalance_metrics(combined_counter)

def pretty_print_metrics(name, m):
    print(f"\n==== Imbalance metrics: {name} ====")
    for k, v in m.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

pretty_print_metrics("Code", metrics_code)
pretty_print_metrics("Subcode", metrics_subcode)
pretty_print_metrics("Combined", metrics_combined)

save_metrics(metrics_code, "metrics_code.json", "metrics_code.csv")
save_metrics(metrics_subcode, "metrics_subcode.json", "metrics_subcode.csv")
save_metrics(metrics_combined, "metrics_combined.json", "metrics_combined.csv")

# ========= Outputs: plots and CSVs =========
# counts CSVs
save_csv(code_counter, "code_level_counts.csv")
save_csv(subcode_counter, "subcode_level_counts.csv")
save_csv(combined_counter, "combined_counts.csv")

# vertical bar plots
plot_bar(code_counter, "Label Counts (Code level)",
         "code_level_counts.png", show_percent=SHOW_PERCENT, top_n=TOP_N)
plot_bar(subcode_counter, "Label Counts (Subcode level)",
         "subcode_level_counts.png", show_percent=SHOW_PERCENT, top_n=TOP_N)
plot_bar(combined_counter, "Label Counts (Code_Subcode combined)",
         "combined_counts.png", show_percent=SHOW_PERCENT, top_n=TOP_N)
