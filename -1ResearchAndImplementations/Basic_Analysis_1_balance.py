"""
Bethesda dataset statistics (fixed alignment):
- Label distributions (code, subcode, combined)
- Distributions across message / sentence / clause using fuzzy matching
- Imbalance metrics

Inputs:
  - /mnt/data/Bethesda_processed_messages_with_annotations.json
  - /mnt/data/Bethesda_messages_with_sentences_and_subsentences.json

Outputs written to /mnt/data/stats_bethesda_fixed/
"""

import json, math, csv
from pathlib import Path
from collections import Counter, defaultdict
from difflib import SequenceMatcher

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------- Paths ----------------
BASE = Path("Bethesda_output")
ANNOT_FILE = BASE / "Bethesda_processed_messages_with_annotations.json"
SPLIT_FILE = BASE / "Bethesda_messages_with_sentences_and_subsentences.json"
OUT_DIR = BASE / "stats_bethesda_fixed"
OUT_DIR.mkdir(exist_ok=True)

# ---------------- Utils ----------------
def compute_metrics(counter: Counter):
    k = len(counter); total = sum(counter.values())
    if not k or not total:
        return {"num_classes": k, "total": total, "IR_max": None, "CV": None,
                "entropy": None, "entropy_norm": None, "gini": None,
                "effective_classes": None, "majority_class_count": None, "minority_class_count": None}
    counts = list(counter.values())
    majority, minority = max(counts), min(counts)
    IR_max = majority / minority if minority else math.inf
    mean = total / k
    std = math.sqrt(sum((c - mean) ** 2 for c in counts) / k)
    CV = std / mean if mean else None
    ps = [c / total for c in counts]
    eps = 1e-12
    entropy = -sum(p * math.log(p + eps) for p in ps)
    entropy_norm = entropy / math.log(k) if k > 1 else 0.0
    gini = 1 - sum(p * p for p in ps)
    eff = 1.0 / sum(p * p for p in ps)
    return {"num_classes": k, "total": total, "IR_max": IR_max, "CV": CV,
            "entropy": entropy, "entropy_norm": entropy_norm,
            "gini": gini, "effective_classes": eff,
            "majority_class_count": majority, "minority_class_count": minority}

def save_counter(counter: Counter, name: str, topn: int = 30):
    csv_path = OUT_DIR / f"{name}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["label", "count"])
        for lbl, cnt in counter.most_common():
            w.writerow([lbl, cnt])
    print(f"[ok] {csv_path}")

    items = counter.most_common(topn)
    if items:
        labels, counts = zip(*items)
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(labels)), counts)
        plt.yticks(range(len(labels)), labels, fontsize=8)
        plt.gca().invert_yaxis()
        plt.title(name)
        plt.tight_layout()
        fig_path = OUT_DIR / f"{name}.png"
        plt.savefig(fig_path, dpi=200); plt.close()
        print(f"[ok] {fig_path}")

def hist_plot(values, name):
    if not values: return
    bins = range(0, max(values)+2)
    plt.figure()
    plt.hist(values, bins=bins)
    plt.title(f"Annotations per {name}")
    plt.xlabel("num annotations"); plt.ylabel("count of units")
    plt.tight_layout()
    out = OUT_DIR / f"hist_{name}.png"
    plt.savefig(out, dpi=200); plt.close()
    print(f"[ok] {out}")

def best_match_idx(text, candidates, threshold=0.60):
    """Return (index, score) of best candidate >= threshold, else (-1, 0)."""
    t = text.lower()
    best_i, best_s = -1, 0.0
    for i, cand in enumerate(candidates):
        s = SequenceMatcher(None, t, cand.lower()).ratio()
        if s > best_s:
            best_s, best_i = s, i
    return (best_i, best_s) if best_s >= threshold else (-1, 0.0)

# ---------------- Load ----------------
ann_data = json.loads(ANNOT_FILE.read_text(encoding="utf-8"))
split_data = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))

# Build dicts by message_id
ann_by_mid = {}                # mid -> list of annotations {text, code}
for rec in ann_data:
    mid = rec["message_id"]
    ann_by_mid[mid] = [{"text": a.get("text",""), "code": a.get("code", [])} for a in rec.get("annotations", [])]

seg_by_mid = {}                # mid -> {sent_texts: [...], clause_texts: [...], sent_ids: [...], clause_ids: [...]}
for rec in split_data:
    mid = rec["message_id"]
    sent_texts, sent_ids = [], []
    clause_texts, clause_ids = [], []
    for s in rec.get("sentences", []):
        sent_texts.append(s.get("sentence","").strip()); sent_ids.append(s.get("sentence_id"))
        for c in s.get("subsentences", []):
            clause_texts.append(c.get("subsentence","").strip()); clause_ids.append(c.get("subsentence_id"))
    seg_by_mid[mid] = {"sent_texts": sent_texts, "clause_texts": clause_texts,
                       "sent_ids": sent_ids, "clause_ids": clause_ids}

# ---------------- 1) Label distributions (annotation-level) ----------------
code_ctr = Counter()
subcode_ctr = Counter()
combined_ctr = Counter()

for rec in ann_data:
    for a in rec.get("annotations", []):
        labels = a.get("code", [])
        if not labels: continue
        code_ctr[labels[0]] += 1
        if len(labels) >= 2:
            subcode_ctr[labels[1]] += 1
            combined_ctr[f"{labels[0]} → {labels[1]}"] += 1
        else:
            combined_ctr[labels[0]] += 1

save_counter(code_ctr, "code_level_counts")
save_counter(subcode_ctr, "subcode_level_counts")
save_counter(combined_ctr, "combined_counts")

imbalance = {
    "code": compute_metrics(code_ctr),
    "subcode": compute_metrics(subcode_ctr),
    "combined": compute_metrics(combined_ctr),
}
(OUT_DIR / "imbalance_metrics.json").write_text(json.dumps(imbalance, indent=2), encoding="utf-8")
print("[ok] imbalance metrics saved")

# ---------------- 2) Distributions by unit with fuzzy alignment ----------------
TH_SENT = 0.60
TH_CLAUSE = 0.60

# per-unit totals
msg_ann_totals = []   # number of annotations per message
sent_ann_totals = []  # number per sentence (across corpus)
clause_ann_totals = []# number per clause

# label frequency by unit type
lbl_by_msg = Counter()
lbl_by_sent = Counter()
lbl_by_clause = Counter()

# iterate messages
for mid, anns in ann_by_mid.items():
    seg = seg_by_mid.get(mid, None)
    if seg is None:
        # still count message-level from annotations alone
        msg_ann_totals.append(len(anns))
        for a in anns:
            for lab in a.get("code", []):
                lbl_by_msg[lab] += 1
        continue

    # message-level: sum of annotations present in this message
    msg_ann_totals.append(len(anns))
    for a in anns:
        for lab in a.get("code", []):
            lbl_by_msg[lab] += 1

    # sentence-level alignment via fuzzy match
    sent_counts_local = [0]*len(seg["sent_texts"])
    for a in anns:
        idx, score = best_match_idx(a["text"], seg["sent_texts"], threshold=TH_SENT)
        if idx >= 0:
            sent_counts_local[idx] += 1
            for lab in a.get("code", []):
                lbl_by_sent[lab] += 1
    sent_ann_totals.extend(sent_counts_local)

    # clause-level alignment via fuzzy match
    clause_counts_local = [0]*len(seg["clause_texts"])
    for a in anns:
        idx, score = best_match_idx(a["text"], seg["clause_texts"], threshold=TH_CLAUSE)
        if idx >= 0:
            clause_counts_local[idx] += 1
            for lab in a.get("code", []):
                lbl_by_clause[lab] += 1
    clause_ann_totals.extend(clause_counts_local)

# save label-by-unit CSVs + plots
save_counter(lbl_by_msg, "labels_by_message")
save_counter(lbl_by_sent, "labels_by_sentence")
save_counter(lbl_by_clause, "labels_by_clause")

# unit histograms
hist_plot(msg_ann_totals, "message")
hist_plot(sent_ann_totals, "sentence")
hist_plot(clause_ann_totals, "clause")

# quick sanity print
print(f"Messages: {len(ann_by_mid)}")
print(f"Sentences (total across corpus): {sum(len(seg_by_mid[m]['sent_texts']) for m in seg_by_mid)}")
print(f"Clauses   (total across corpus): {sum(len(seg_by_mid[m]['clause_texts']) for m in seg_by_mid)}")
