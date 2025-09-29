"""
Count label coverage per message and per annotation, and create visualizations.

Inputs:
  - /mnt/data/Bethesda_processed_messages_with_annotations.json
  - /mnt/data/codebook_hierarchy.json

Outputs:
  - /mnt/data/Bethesda_label_counts.json
  - /mnt/data/fig_code_level_topN.png
  - /mnt/data/fig_subcode_level_topN.png
  - /mnt/data/fig_heatmap_L1xL2.png
  - (optional) annotation-level figures if ANNOTATION_LEVEL_PLOTS = True
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
import math

# --------------- Paths and settings ---------------
MSG_PATH  = Path("Bethesda_output/Bethesda_processed_messages_with_annotations.json")
HIER_PATH = Path("PV_output_json/codebook_hierarchy.json")
OUT_JSON  = Path("Bethesda_output/Bethesda_label_counts.json")

# Figure outputs
FIG_CODE_TOP   = Path("Bethesda_output/fig_code_level_topN.png")
FIG_SUBCODE_TOP= Path("Bethesda_output/fig_subcode_level_topN.png")
FIG_HEATMAP    = Path("Bethesda_output/fig_heatmap_L1xL2.png")

TOPN_CODE      = 15
TOPN_SUBCODE   = 20
ANNOTATION_LEVEL_PLOTS = False   # switch to True if you also want annotation-level charts

# --------------- Matplotlib setup ---------------
import matplotlib
# use non-interactive backend when running headless
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def barh_plot(pairs, title, outfile, xlabel="Messages", figsize=(9, 6)):
    """
    pairs: list[(label, count)]
    """
    if not pairs:
        print(f"[warn] No data for {title}")
        return
    labels = [p[0] for p in pairs]
    counts = [p[1] for p in pairs]
    idx = np.arange(len(labels))
    plt.figure(figsize=figsize)
    plt.barh(idx, counts)
    plt.yticks(idx, labels, fontsize=9)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.gca().invert_yaxis()  # largest at top
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()
    print(f"[ok] Saved {outfile}")

def heatmap(matrix, rows, cols, title, outfile, figsize=(10, 7)):
    if matrix.size == 0:
        print(f"[warn] Empty heatmap for {title}")
        return
    plt.figure(figsize=figsize)
    im = plt.imshow(matrix, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Messages with pair")
    plt.xticks(np.arange(len(cols)), cols, rotation=45, ha="right", fontsize=9)
    plt.yticks(np.arange(len(rows)), rows, fontsize=9)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=250)
    plt.close()
    print(f"[ok] Saved {outfile}")

# --------------- Normalization helpers ---------------
def norm(s: str) -> str:
    return (
        s.replace("-", "_")
         .replace(" ", "_")
         .replace("&", "and")
         .replace("__", "_")
         .strip()
         .lower()
    )

# You can extend this alias table if you see new variants
ALIAS = {
    # Top level
    "sdoh": "sdoh",
    "partnershippatient": "partenership_form_the_patient_side",
    "partnershipprovider": "partnership_from_the_provider_side",
    "shareddecisionpatient": "shared-decision_making_from_the_patient_side",
    "shareddecisionmakingfromthepatientside": "shared-decision_making_from_the_patient_side",
    "shareddecisionmakingfromtheproviderside": "shared-decision_making_form_the_provider_side",
    "carecoordinationpatient": "care_coordination_patient",
    "carecoordinationprovider": "care_coordination_provider",

    # SDOH subcodes
    "economicstability": "economic_stability",
    "educationaccessandquality": "education_access_and_quality",
    "healthcareaccessandquality": "health_care_access_and_quality",
    "neighborhoodandbuiltenvironment": "neighborhood_and_built_environment",
    "socialandcommunitycontext": "social_and_community_context",

    # SDOH subsubcodes (examples)
    "employment_status": "employment_status",
    "work-related_challenges": "work-related_challenges",
    "financial_insecurity": "financial_insecurity",
    "insurance": "insurance",
    "health_access": "health_access",
    "diet": "diet",
    "sleep": "sleep",
    "transportation": "transportation",
    "environmental_factors": "environmental_factors",
    "weather": "weather",
    "level_of_education": "level_of_education",
    "social_support": "social_support",

    # Partnership (patient side) subcodes
    "activeparticipation/involvement": "active_participation_/involvement",
    "expressopinions": "express_opinions",
    "statepreferences": "state_preferences",
    "appreciation/gratitude": "appreciation/gratitude",
    "connection": "connection",
    "salutation": "salutation",
    "clinical_care": "clinical_care",
    "build_trust": "build_trust",

    # Partnership (provider side) subcodes
    "requestsforopinion": "requests_for_opinion",
    "requestsfropinion": "requests_for_opinion",
    "invite_collaboration": "invite_collaboration",
    "checking_understanding/clarification": "checking_understanding/clarification",
    "acknowledge_patient_expertise/knowledge": "acknowledge_patient_expertise/knowledge",
    "maintain_communication": "maintain_communication",

    # Shared decision
    "exploreoptions": "explore_options",
    "seekingapproval": "seeking_approval",
    "approvalofdecision": "approval_of_decision",
    "approval_of_decision/reinforcement": "approval_of_decision/reinforcement",
    "share_options": "share_options",
    "summarize_and_confirm_understanding": "summarize_and_confirm_understanding",
    "make_decision": "make_decision",
}

def alias_or_self(label: str) -> str:
    n = norm(label)
    return ALIAS.get(n, n)

# --------------- Load hierarchy ---------------
hier = json.loads(HIER_PATH.read_text(encoding="utf-8"))

level_ids = {"code": set(), "subcode": set(), "subsubcode": set()}
id_map_norm_to_true = {}

for node in hier.get("nodes", []):
    true_id = node.get("id", "")
    t = node.get("type", "").lower()
    nkey = norm(true_id)
    id_map_norm_to_true[nkey] = true_id
    if t in level_ids:
        level_ids[t].add(nkey)

# small robustness for provider-side partnership spelling
if "partnership_from_the_provider_side" not in level_ids["code"]:
    for k, v in id_map_norm_to_true.items():
        if v.lower().startswith("partner") and "provider" in v.lower():
            level_ids["code"].add(k)

# --------------- Counting ---------------
data = json.loads(MSG_PATH.read_text(encoding="utf-8"))

# per-message presence
msg_by_code    = defaultdict(set)    # L1
msg_by_subcode = defaultdict(set)    # L2
msg_by_subsub  = defaultdict(set)    # L3
msg_by_L1L2    = defaultdict(set)    # (L1,L2)
msg_by_L1L2L3  = defaultdict(set)    # (L1,L2,L3)

# per-annotation frequency
ann_code    = Counter()
ann_subcode = Counter()
ann_subsub  = Counter()
ann_L1L2    = Counter()
ann_L1L2L3  = Counter()

unmapped = set()

for rec in data:
    mid = rec["message_id"]
    for ann in rec.get("annotations", []):
        raw = ann.get("code", []) or []
        labels = [alias_or_self(x) for x in raw]
        L1 = L2 = L3 = None

        if labels:
            c1 = labels[0]
            if c1 in level_ids["code"]:
                L1 = c1
            else:
                unmapped.add(raw[0])
        if len(labels) >= 2:
            c2 = labels[1]
            if c2 in level_ids["subcode"]:
                L2 = c2
            elif c2 not in level_ids["code"]:
                unmapped.add(raw[1])
        if len(labels) >= 3:
            c3 = labels[2]
            if c3 in level_ids["subsubcode"]:
                L3 = c3
            else:
                unmapped.add(raw[2])

        # presence per message
        if L1: msg_by_code[L1].add(mid)
        if L2: msg_by_subcode[L2].add(mid)
        if L3: msg_by_subsub[L3].add(mid)
        if L1 and L2: msg_by_L1L2[(L1, L2)].add(mid)
        if L1 and L2 and L3: msg_by_L1L2L3[(L1, L2, L3)].add(mid)

        # frequency per annotation
        if L1: ann_code[L1] += 1
        if L2: ann_subcode[L2] += 1
        if L3: ann_subsub[L3] += 1
        if L1 and L2: ann_L1L2[(L1, L2)] += 1
        if L1 and L2 and L3: ann_L1L2L3[(L1, L2, L3)] += 1

# --------------- Serialize counts ---------------
def map_back_label(k):
    return id_map_norm_to_true.get(k, k)

def map_back_tuple(tpl):
    return tuple(id_map_norm_to_true.get(x, x) for x in tpl)

def to_counts(d, tuple_keys=False):
    if tuple_keys:
        return [
            {"path": map_back_tuple(k), "message_count": len(v)}
            for k, v in sorted(d.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        ]
    else:
        return [
            {"label": map_back_label(k), "message_count": len(v)}
            for k, v in sorted(d.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        ]

def to_counts_ann(cntr, tuple_keys=False):
    if tuple_keys:
        return [
            {"path": map_back_tuple(k), "annotation_count": v}
            for k, v in cntr.most_common()
        ]
    else:
        return [
            {"label": map_back_label(k), "annotation_count": v}
            for k, v in cntr.most_common()
        ]

results = {
    "counts": {
        "message_level": {
            "code_level": to_counts(msg_by_code),
            "subcode_level": to_counts(msg_by_subcode),
            "subsubcode_level": to_counts(msg_by_subsub),
            "combined_L1_L2": to_counts(msg_by_L1L2, tuple_keys=True),
            "combined_L1_L2_L3": to_counts(msg_by_L1L2L3, tuple_keys=True),
        },
        "annotation_level": {
            "code_level": to_counts_ann(ann_code),
            "subcode_level": to_counts_ann(ann_subcode),
            "subsubcode_level": to_counts_ann(ann_subsub),
            "combined_L1_L2": to_counts_ann(ann_L1L2, tuple_keys=True),
            "combined_L1_L2_L3": to_counts_ann(ann_L1L2L3, tuple_keys=True),
        }
    },
    "meta": {
        "total_messages": len(data),
        "unmapped_label_examples": sorted(list(unmapped))[:30]
    }
}

OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[ok] Wrote {OUT_JSON}")

# --------------- Build plot inputs (message-level) ---------------
top_code = [(row["label"], row["message_count"])
            for row in results["counts"]["message_level"]["code_level"][:TOPN_CODE]]

top_subcode = [(row["label"], row["message_count"])
               for row in results["counts"]["message_level"]["subcode_level"][:TOPN_SUBCODE]]

# heatmap of L1 x L2 message coverage
L1_labels = [row["label"] for row in results["counts"]["message_level"]["code_level"]]
L2_labels = [row["label"] for row in results["counts"]["message_level"]["subcode_level"]]

# keep top K rows/cols to keep figure readable
max_rows = min(20, len(L1_labels))
max_cols = min(30, len(L2_labels))
L1_keep = L1_labels[:max_rows]
L2_keep = L2_labels[:max_cols]

pair_counts = {tuple(row["path"]): row["message_count"]
               for row in results["counts"]["message_level"]["combined_L1_L2"]}

mat = np.zeros((len(L1_keep), len(L2_keep)), dtype=float)
for i, l1 in enumerate(L1_keep):
    for j, l2 in enumerate(L2_keep):
        key = (l1, l2)
        mat[i, j] = pair_counts.get(key, 0)

# --------------- Plot and save ---------------
barh_plot(top_code,
          title=f"Top {TOPN_CODE} codes by message coverage",
          outfile=str(FIG_CODE_TOP),
          xlabel="Messages")

barh_plot(top_subcode,
          title=f"Top {TOPN_SUBCODE} subcodes by message coverage",
          outfile=str(FIG_SUBCODE_TOP),
          xlabel="Messages")

heatmap(mat,
        rows=L1_keep,
        cols=L2_keep,
        title="Message coverage heatmap: Code (rows) × Subcode (cols)",
        outfile=str(FIG_HEATMAP))

# --------------- Optional: annotation-level versions ---------------
if ANNOTATION_LEVEL_PLOTS:
    FIG_CODE_TOP_ANN = Path("Bethesda_output/fig_code_level_topN_ANN.png")
    FIG_SUBCODE_TOP_ANN = Path("Bethesda_output/fig_subcode_level_topN_ANN.png")
    FIG_HEATMAP_ANN = Path("Bethesda_output/fig_heatmap_L1xL2_ANN.png")

    top_code_ann = [(row["label"], row["annotation_count"])
                    for row in results["counts"]["annotation_level"]["code_level"][:TOPN_CODE]]
    top_subcode_ann = [(row["label"], row["annotation_count"])
                       for row in results["counts"]["annotation_level"]["subcode_level"][:TOPN_SUBCODE]]

    pair_ann = {tuple(row["path"]): row["annotation_count"]
                for row in results["counts"]["annotation_level"]["combined_L1_L2"]}

    mat_ann = np.zeros((len(L1_keep), len(L2_keep)), dtype=float)
    for i, l1 in enumerate(L1_keep):
        for j, l2 in enumerate(L2_keep):
            key = (l1, l2)
            mat_ann[i, j] = pair_ann.get(key, 0)

    barh_plot(top_code_ann,
              title=f"Top {TOPN_CODE} codes by annotation frequency",
              outfile=str(FIG_CODE_TOP_ANN),
              xlabel="Annotations")

    barh_plot(top_subcode_ann,
              title=f"Top {TOPN_SUBCODE} subcodes by annotation frequency",
              outfile=str(FIG_SUBCODE_TOP_ANN),
              xlabel="Annotations")

    heatmap(mat_ann,
            rows=L1_keep,
            cols=L2_keep,
            title="Annotation frequency heatmap: Code (rows) × Subcode (cols)",
            outfile=str(FIG_HEATMAP_ANN))
