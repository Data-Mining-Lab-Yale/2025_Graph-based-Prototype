# Dataset_Stat_3_LabelAmbiguity.py
# Inputs:
#   - processed_messages_with_annotations.json
#   - annotation_code_mapping_detailed_corrected.json
# Outputs (in ambiguity_stats/):
#   - nn_conflict_samples.csv  (anchor, neighbor, labels, similarity)
#   - ambiguous_label_pairs.csv (LabelA, LabelB, count, avg_similarity)
#   - per_label_disagreement.csv (label, avg_cross_label_rate)
#   - summary.json (global metrics)
#   - cluster_purity.json

import json, os, math, csv, random
from collections import defaultdict, Counter

import numpy as np

# Try sentence-transformers first, else fall back to TF-IDF
USE_TFIDF_FALLBACK = False
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    USE_TFIDF_FALLBACK = True

if USE_TFIDF_FALLBACK:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

from sklearn.cluster import KMeans


# ========= Config (extend) =========
# options: "subcode" | "code" | "combined"
LABEL_LEVEL = "combined"
SUFFIX_IN_OUTDIR = True
REQUIRE_BOTH_FOR_COMBINED = True   # if True, skip samples missing code or subcode


# ========= Config =========
ANNOTATION_FILE = "EPPC_output_json/CleanedData/processed_messages_with_annotations.json"
MAPPING_FILE    = "EPPC_output_json/Labels/annotation_code_mapping_detailed_corrected.json"
# OUTDIR          = "Data_for_Evidences/ambiguity_stats_combined"
# ========= I/O dir handling (unchanged except suffix) =========
OUTDIR = "Data_for_Evidences/ambiguity_stats"
if SUFFIX_IN_OUTDIR:
    OUTDIR = f"{OUTDIR}_{LABEL_LEVEL}"
os.makedirs(OUTDIR, exist_ok=True)

# choose which text field to use when available
POSSIBLE_TEXT_KEYS = [
    "span_text", "clause_text", "segment_text", "text", "content"
]

# nearest neighbor params
TOP_K = 5                 # neighbors per item
SIM_THRESHOLD = 0.55      # only count conflicts above this cosine
MAX_EXAMPLE_ROWS = 200    # limit csv size of example pairs

random.seed(123)
np.random.seed(123)

os.makedirs(OUTDIR, exist_ok=True)

# ========= Load =========
with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    messages = json.load(f)
with open(MAPPING_FILE, "r", encoding="utf-8") as f:
    mapping = json.load(f)

# ========= Helpers =========
def lookup_token(tok: str):
    if tok in mapping: return mapping[tok]
    t_low, t_up = tok.lower(), tok.upper()
    if t_low in mapping: return mapping[t_low]
    if t_up in mapping:  return mapping[t_up]
    return None

def get_ann_text(ann: dict):
    for k in POSSIBLE_TEXT_KEYS:
        if k in ann and isinstance(ann[k], str) and ann[k].strip():
            return ann[k].strip()
    # fallback: try to compose from message text and offsets if present
    if "text" in ann and isinstance(ann["text"], str):
        return ann["text"].strip()
    return None

# ========= Replace choose_label with the two helpers below =========
# ========= Helpers (add/replace) =========
def extract_label_levels(ann: dict):
    """
    Returns {"code": <str or None>, "subcode": <str or None>}
    based on your token->codebook lookup.
    """
    got = {"code": None, "subcode": None}
    raw = ann.get("code", [])
    if not isinstance(raw, list):
        return got
    for tok in raw:
        info = lookup_token(tok)  # your existing function
        if not info:
            continue
        lvl = info.get("level")
        lab = info.get("matched_codebook_label")
        if lvl in got and got[lvl] is None and lab:
            got[lvl] = lab
    return got

def choose_label_by_level(ann: dict, level: str):
    """
    Strict single-level selection for "code" or "subcode".
    """
    lv = extract_label_levels(ann)
    return lv.get(level)


def choose_label(ann: dict):
    """
    Backward compatible: prefer subcode, else code.
    Kept for reference. Not used when LABEL_LEVEL is set.
    """
    raw = ann.get("code", [])
    if not isinstance(raw, list):
        return None
    code_label, subcode_label = None, None
    for tok in raw:
        info = lookup_token(tok)
        if not info:
            continue
        lvl = info.get("level")
        matched = info.get("matched_codebook_label")
        if lvl == "code" and code_label is None:
            code_label = matched
        elif lvl == "subcode" and subcode_label is None:
            subcode_label = matched
    return subcode_label if subcode_label else code_label


def make_combined_label(code_label: str | None, subcode_label: str | None):
    """
    Build a stable combined label string. Use a delimiter that never appears
    in your labels. If unsure, '||' is usually safe.
    """
    if code_label is None and subcode_label is None:
        return None
    if REQUIRE_BOTH_FOR_COMBINED and (code_label is None or subcode_label is None):
        return None
    # fallbacks when you allow missing parts
    cl = code_label if code_label is not None else "∅"
    sl = subcode_label if subcode_label is not None else "∅"
    return f"{cl}||{sl}"



# ========= Build dataset of clauses (small change here) =========
# ========= Build dataset (replace your current loop) =========
texts, labels = [], []
for msg in messages:
    for ann in msg.get("annotations", []):
        t = get_ann_text(ann)  # your existing function
        if not t:
            continue

        if LABEL_LEVEL in ("code", "subcode"):
            y = choose_label_by_level(ann, LABEL_LEVEL)
        elif LABEL_LEVEL == "combined":
            lv = extract_label_levels(ann)
            y = make_combined_label(lv["code"], lv["subcode"])
        else:
            raise ValueError(f"Unknown LABEL_LEVEL: {LABEL_LEVEL}")

        if y:
            texts.append(t)
            labels.append(y)

# ========= Optional: write a small legend to decode combined labels =========
legend_path = os.path.join(OUTDIR, "combined_label_legend.txt")
if LABEL_LEVEL == "combined":
    uniq = sorted(set(labels))
    with open(legend_path, "w", encoding="utf-8") as f:
        f.write("Format: CODE||SUBCODE\n")
        for k in uniq:
            f.write(k + "\n")

# ========= When writing outputs, include level in filenames for clarity =========
# Example: change the three writers' target filenames:
#   "nn_conflict_samples.csv" -> f"nn_conflict_samples_{LABEL_LEVEL}.csv" if you prefer
# Or keep as-is since OUTDIR already has the suffix.

N = len(texts)
unique_labels = sorted(set(labels))
print(f"[INFO] Clauses with usable text+label: {N}")
print(f"[INFO] Unique labels used for ambiguity checks: {len(unique_labels)}")

# ========= Embeddings =========
if not USE_TFIDF_FALLBACK:
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        X = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
        # cosine_similarity on normalized embeddings equals dot product, but we call it for clarity
        print("[INFO] Using SentenceTransformer embeddings.")
    except Exception as e:
        print(f"[WARN] SentenceTransformer failed: {e}")
        USE_TFIDF_FALLBACK = True

if USE_TFIDF_FALLBACK:
    print("[INFO] Using TF-IDF fallback.")
    vec = TfidfVectorizer(min_df=2, ngram_range=(1,2), lowercase=True)
    X = vec.fit_transform(texts)

# ========= Nearest neighbors and conflict stats =========
# Compute cosine similarity matrix in blocks to avoid large memory use.
# For N ~ 2.6k, full matrix is OK, but we keep it simple.

if USE_TFIDF_FALLBACK:
    sim = cosine_similarity(X)
else:
    # For normalized ST embeddings, cosine equals dot product
    sim = np.matmul(X, X.T)

# zero out self similarities
np.fill_diagonal(sim, -1.0)

# For each item, pick top-k neighbors
topk_idx = np.argpartition(sim, -TOP_K, axis=1)[:, -TOP_K:]
# sort those top-k by actual similarity
row_sorted = np.take_along_axis(sim, topk_idx, axis=1)
order = np.argsort(-row_sorted, axis=1)
topk_idx_sorted = np.take_along_axis(topk_idx, order, axis=1)
topk_sim_sorted = np.take_along_axis(row_sorted, order, axis=1)

# Measure cross-label neighbor rate and collect conflict pairs
conflict_pairs = Counter()
per_item_cross = []
examples = []

for i in range(N):
    yi = labels[i]
    sims = topk_sim_sorted[i]
    idxs = topk_idx_sorted[i]
    cross = 0
    total_considered = 0
    for s, j in zip(sims, idxs):
        if s < SIM_THRESHOLD:
            continue
        total_considered += 1
        yj = labels[j]
        # record conflicts
        if yj != yi:
            cross += 1
            a, b = sorted([yi, yj])
            conflict_pairs[(a, b)] += 1
            if len(examples) < MAX_EXAMPLE_ROWS:
                examples.append([texts[i], yi, texts[j], yj, float(s)])
    rate = cross / total_considered if total_considered > 0 else 0.0
    per_item_cross.append(rate)

global_cross_rate = float(np.mean(per_item_cross))
median_cross_rate = float(np.median(per_item_cross))
print(f"[INFO] Mean cross-label neighbor rate (k={TOP_K}, thr={SIM_THRESHOLD}): {global_cross_rate:.4f}")
print(f"[INFO] Median cross-label neighbor rate: {median_cross_rate:.4f}")

# Save example conflicts
with open(os.path.join(OUTDIR, "nn_conflict_samples.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["anchor_text", "anchor_label", "neighbor_text", "neighbor_label", "similarity"])
    for row in examples:
        w.writerow(row)
print("[OK] Saved nn_conflict_samples.csv")

# Summarize top ambiguous label pairs
pair_rows = []
for (a, b), cnt in conflict_pairs.most_common():
    # estimate average similarity for this pair by sampling up to 200 pairs from examples or recompute quickly
    pair_rows.append([a, b, cnt])
with open(os.path.join(OUTDIR, "ambiguous_label_pairs.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["label_A", "label_B", "count"])
    for r in pair_rows:
        w.writerow(r)
print("[OK] Saved ambiguous_label_pairs.csv")

# Per-label average cross rate
per_label_rates = defaultdict(list)
for r, y in zip(per_item_cross, labels):
    per_label_rates[y].append(r)
with open(os.path.join(OUTDIR, "per_label_disagreement.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["label", "avg_cross_label_rate", "n_items"])
    for lab in sorted(per_label_rates.keys()):
        vals = per_label_rates[lab]
        w.writerow([lab, float(np.mean(vals)), len(vals)])
print("[OK] Saved per_label_disagreement.csv")

# ========= Clustering purity =========
k = len(unique_labels)
k = max(k, 2)
if USE_TFIDF_FALLBACK:
    # KMeans expects dense or CSR ok. For speed, use default.
    km = KMeans(n_clusters=k, n_init=10, random_state=123)
    km.fit(X)
else:
    km = KMeans(n_clusters=k, n_init=10, random_state=123)
    km.fit(X)

cluster_ids = km.labels_
# Purity = sum over clusters of max label count in that cluster divided by N
cluster_to_labels = defaultdict(list)
for cid, y in zip(cluster_ids, labels):
    cluster_to_labels[cid].append(y)
majority_sum = 0
for cid, lst in cluster_to_labels.items():
    majority_sum += Counter(lst).most_common(1)[0][1]
purity = majority_sum / N

with open(os.path.join(OUTDIR, "cluster_purity.json"), "w", encoding="utf-8") as f:
    json.dump({"k": k, "purity": purity, "N": N}, f, indent=2)
print(f"[INFO] Cluster purity (k={k}): {purity:.4f}")
print("[OK] Saved cluster_purity.json")

# ========= Summary =========
summary = {
    "N": N,
    "num_labels": len(unique_labels),
    "mean_cross_label_neighbor_rate": global_cross_rate,
    "median_cross_label_neighbor_rate": median_cross_rate,
    "top_k_neighbors": TOP_K,
    "similarity_threshold": SIM_THRESHOLD,
    "embedding": "tfidf" if USE_TFIDF_FALLBACK else "all-MiniLM-L6-v2",
}
with open(os.path.join(OUTDIR, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print("[OK] Saved summary.json")
