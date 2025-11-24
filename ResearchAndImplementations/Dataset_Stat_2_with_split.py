# Dataset_Stat_2_Split_Distributions.py
import json, os, math, csv
from collections import Counter
import matplotlib.pyplot as plt

# ========= Inputs (edit if needed) =========
ANNOTATION_FILE = "EPPC_output_json/CleanedData/processed_messages_with_annotations.json"
MAPPING_FILE    = "EPPC_output_json/Labels/annotation_code_mapping_detailed_corrected.json"
SPLIT_FILE      = "EPPC_output_json/Labels/split_intents_by_type.json"
OUTDIR          = "Data_for_Evidences/split_stats"   # outputs saved here

TOP_N = None          # set to int to show only top-N in plots
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
    if token in mapping: return mapping[token]
    t_low, t_up = token.lower(), token.upper()
    if t_low in mapping: return mapping[t_low]
    if t_up in mapping:  return mapping[t_up]
    return None

def compute_imbalance_metrics(counter: Counter):
    k = len(counter)
    total = sum(counter.values())
    if k == 0 or total == 0:
        return {
            "num_classes": k, "total": total,
            "IR_max": None, "CV": None,
            "entropy": None, "entropy_normalized": None,
            "gini": None, "effective_classes": None,
            "majority_class_count": None, "minority_class_count": None
        }
    counts = list(counter.values())
    majority, minority = max(counts), min(counts)
    IR_max = majority / minority if minority > 0 else math.inf
    mean = total / k
    var  = sum((c - mean)**2 for c in counts) / k
    std  = math.sqrt(var)
    CV   = std / mean if mean > 0 else None
    ps = [c/total for c in counts]
    eps = 1e-12
    H   = -sum(p*math.log(p + eps) for p in ps)
    Hn  = H / math.log(k) if k > 1 else 0.0
    G   = 1.0 - sum(p*p for p in ps)
    Ke  = 1.0 / sum(p*p for p in ps)
    return {
        "num_classes": k, "total": total,
        "IR_max": IR_max, "CV": CV,
        "entropy": H, "entropy_normalized": Hn,
        "gini": G, "effective_classes": Ke,
        "majority_class_count": majority, "minority_class_count": minority
    }

def plot_bar(counter: Counter, title: str, filename: str, show_percent=False, top_n=None):
    if not counter:
        print(f"[WARN] No data to plot for: {title}")
        return
    items = counter.most_common(top_n) if top_n else counter.most_common()
    labels, counts = zip(*items)
    if show_percent:
        total = sum(counts)
        values = [c/total*100 for c in counts]; ylabel = "Percentage (%)"
    else:
        values = counts; ylabel = "Count"
    plt.figure(figsize=(max(8, 0.45*len(labels)), 6))
    plt.bar(labels, values)
    plt.xticks(rotation=60, ha="right")
    plt.ylabel(ylabel); plt.title(title)
    plt.tight_layout(); plt.savefig(filename, dpi=DPI); plt.close()
    print(f"[OK] Saved: {filename}")

def save_counts(counter: Counter, filename: str):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["label","count"])
        for lab, cnt in counter.most_common(): w.writerow([lab, cnt])
    print(f"[OK] Saved: {filename}")

def save_metrics(metrics: dict, filename_json: str, filename_csv: str):
    with open(filename_json, "w", encoding="utf-8") as f: json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(filename_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["metric", "value"])
        for k,v in metrics.items(): w.writerow([k, v])
    print(f"[OK] Saved: {filename_json}, {filename_csv}")

# Build sets of labels for each split and level
def build_label_sets(split_block):
    codes, subcodes = set(), set()
    for item in split_block:
        lvl = item.get("level"); lab = item.get("label")
        if not lvl or not lab: continue
        if lvl.lower() == "code":    codes.add(lab)
        elif lvl.lower() == "subcode": subcodes.add(lab)
        # subsubcodes not used for these counts; add if you want
    return codes, subcodes

INTER_codes, INTER_subcodes = build_label_sets(split.get("Interactional", []))
GOAL_codes,  GOAL_subcodes  = build_label_sets(split.get("Goal-Oriented", []))

def process_split(split_name, allowed_codes: set, allowed_subcodes: set):
    code_ctr = Counter(); subcode_ctr = Counter(); combined_ctr = Counter()
    total_items = 0; mapped_items = 0
    for msg in messages:
        for ann in msg.get("annotations", []):
            total_items += 1
            raw_tokens = ann.get("code", [])
            if not isinstance(raw_tokens, list): continue
            code_label, subcode_label = None, None
            for tok in raw_tokens:
                info = lookup_token(tok)
                if not info: continue
                lvl = info.get("level"); matched = info.get("matched_codebook_label")
                if lvl == "code" and code_label is None: code_label = matched
                elif lvl == "subcode" and subcode_label is None: subcode_label = matched
            if code_label or subcode_label: mapped_items += 1

            # Count per level restricted to this split
            if code_label and code_label in allowed_codes:
                code_ctr[code_label] += 1
            if subcode_label and subcode_label in allowed_subcodes:
                subcode_ctr[subcode_label] += 1

            # Combined: prefer true pairs whose subcode is inside the split.
            if code_label and subcode_label and subcode_label in allowed_subcodes:
                combined_ctr[f"{code_label}_{subcode_label}"] += 1
            # If no subcode but code is allowed, fall back to code label as combined entry.
            elif subcode_label is None and code_label and code_label in allowed_codes:
                combined_ctr[code_label] += 1
            # If only subcode present and allowed, count the subcode alone.
            elif code_label is None and subcode_label and subcode_label in allowed_subcodes:
                combined_ctr[subcode_label] += 1

    print(f"\n===== {split_name} =====")
    print(f"[INFO] Total annotation items: {total_items}")
    print(f"[INFO] Items with at least one mapped token: {mapped_items}")

    # Metrics
    m_code     = compute_imbalance_metrics(code_ctr)
    m_subcode  = compute_imbalance_metrics(subcode_ctr)
    m_combined = compute_imbalance_metrics(combined_ctr)

    # Save counts
    save_counts(code_ctr,     os.path.join(OUTDIR, f"{split_name.lower()}_code_counts.csv"))
    save_counts(subcode_ctr,  os.path.join(OUTDIR, f"{split_name.lower()}_subcode_counts.csv"))
    save_counts(combined_ctr, os.path.join(OUTDIR, f"{split_name.lower()}_combined_counts.csv"))

    # Save metrics
    save_metrics(m_code,     os.path.join(OUTDIR, f"{split_name.lower()}_metrics_code.json"),
                           os.path.join(OUTDIR, f"{split_name.lower()}_metrics_code.csv"))
    save_metrics(m_subcode,  os.path.join(OUTDIR, f"{split_name.lower()}_metrics_subcode.json"),
                           os.path.join(OUTDIR, f"{split_name.lower()}_metrics_subcode.csv"))
    save_metrics(m_combined, os.path.join(OUTDIR, f"{split_name.lower()}_metrics_combined.json"),
                           os.path.join(OUTDIR, f"{split_name.lower()}_metrics_combined.csv"))

    # Plots
    plot_bar(code_ctr,     f"{split_name}: Label Counts (Code level)",
             os.path.join(OUTDIR, f"{split_name.lower()}_code_counts.png"),
             show_percent=SHOW_PERCENT, top_n=TOP_N)
    plot_bar(subcode_ctr,  f"{split_name}: Label Counts (Subcode level)",
             os.path.join(OUTDIR, f"{split_name.lower()}_subcode_counts.png"),
             show_percent=SHOW_PERCENT, top_n=TOP_N)
    plot_bar(combined_ctr, f"{split_name}: Label Counts (Code_Subcode combined)",
             os.path.join(OUTDIR, f"{split_name.lower()}_combined_counts.png"),
             show_percent=SHOW_PERCENT, top_n=TOP_N)

    # Pretty print
    def pp(tag, m):
        print(f"\n-- {split_name} {tag} --")
        for k,v in m.items():
            if isinstance(v, float): print(f"{k}: {v:.6f}")
            else: print(f"{k}: {v}")
    pp("Code metrics", m_code)
    pp("Subcode metrics", m_subcode)
    pp("Combined metrics", m_combined)

# ========= Run for both splits =========
process_split("Interactional", INTER_codes, INTER_subcodes)
process_split("GoalOriented",  GOAL_codes,  GOAL_subcodes)
print("\n[DONE] Outputs in:", OUTDIR)
