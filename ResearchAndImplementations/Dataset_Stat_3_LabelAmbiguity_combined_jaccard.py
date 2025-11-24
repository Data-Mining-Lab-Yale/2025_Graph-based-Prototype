#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ambiguity analysis with configurable semantic representations.

Preserves your original pipeline:
  - same text field preference order
  - same neighbor selection and counting
  - same outputs plus enhanced pair stats and label totals

Representations:
  - sbert   : pretrained sentence embeddings + cosine
  - tfidf   : TF-IDF bag of words + cosine
  - lsa     : TF-IDF -> TruncatedSVD (LSA) + cosine
  - jaccard : token sets + Jaccard similarity
"""

import os, json, csv, random
import numpy as np
from collections import defaultdict, Counter

# ==============================
# CONFIG: edit here
# ==============================
ANNOTATION_FILE = "EPPC_output_json/CleanedData/processed_messages_with_annotations.json"
MAPPING_FILE    = "EPPC_output_json/Labels/annotation_code_mapping_detailed_corrected.json"

# Output base directory
OUTDIR_BASE = "Data_for_Evidences/ambiguity_stats_jaccard"

# Label level: "code" | "subcode" | "combined"
LABEL_LEVEL = "combined"
REQUIRE_BOTH_FOR_COMBINED = True

# Semantic representation: "sbert" | "tfidf" | "lsa" | "jaccard"
REP_KIND = "jaccard"
SBERT_MODEL = "all-MiniLM-L6-v2"   # used when REP_KIND == "sbert"

# Neighbor params
TOP_K = 5
SIM_THRESHOLD = 0.55
MAX_EXAMPLE_ROWS = 200

# Which text field to use in order of preference
POSSIBLE_TEXT_KEYS = ["span_text", "clause_text", "segment_text", "text", "content"]

# Random seeds for reproducibility
random.seed(123)
np.random.seed(123)
# ==============================


# ---------- IO helpers ----------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------- Label mapping helpers ----------
def lookup_token(tok, mapping):
    if tok in mapping:
        return mapping[tok]
    t_low, t_up = tok.lower(), tok.upper()
    if t_low in mapping:
        return mapping[t_low]
    if t_up in mapping:
        return mapping[t_up]
    return None

def extract_label_levels(ann, mapping):
    got = {"code": None, "subcode": None}
    raw = ann.get("code", [])
    if not isinstance(raw, list):
        return got
    for tok in raw:
        info = lookup_token(tok, mapping)
        if not info:
            continue
        lvl = info.get("level")
        lab = info.get("matched_codebook_label")
        if lvl in got and got[lvl] is None and lab:
            got[lvl] = lab
    return got

def choose_label_by_level(ann, level, mapping):
    lv = extract_label_levels(ann, mapping)
    return lv.get(level)

def make_combined_label(code_label, subcode_label, require_both=True):
    if code_label is None and subcode_label is None:
        return None
    if require_both and (code_label is None or subcode_label is None):
        return None
    cl = code_label if code_label is not None else "∅"
    sl = subcode_label if subcode_label is not None else "∅"
    return f"{cl}||{sl}"


# ---------- Text selection ----------
def get_ann_text(ann):
    for k in POSSIBLE_TEXT_KEYS:
        if k in ann and isinstance(ann[k], str) and ann[k].strip():
            return ann[k].strip()
    if "text" in ann and isinstance(ann["text"], str):
        return ann["text"].strip()
    return None


# ---------- Build representations ----------
def build_repr_and_sim(texts, rep_kind, sbert_model="all-MiniLM-L6-v2"):
    """
    Returns (features, similarity_matrix, rep_name, sim_name)

    Modes
      - sbert   : pretrained sentence embeddings + cosine  [GENERAL semantic]
      - tfidf   : TF-IDF bag of words + cosine             [DATASET specific]
      - lsa     : TF-IDF -> TruncatedSVD + cosine          [DATASET specific]
      - jaccard : binary bag-of-words + Jaccard similarity [lexical overlap]
    """
    rep_kind = rep_kind.lower()

    if rep_kind == "sbert":
        # General semantic. Independent from your dataset vocabulary and idf.
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(sbert_model)
        X = model.encode(
            texts, batch_size=64, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True
        )
        sim = np.matmul(X, X.T)  # cosine since vectors are L2 normalized
        print("[REP] SBERT embeddings  | [SIM] cosine  | semantic=GENERAL")
        return X, sim, sbert_model, "cosine"

    elif rep_kind == "tfidf":
        # Dataset specific. Vocabulary and idf come from your corpus.
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vec = TfidfVectorizer(min_df=2, ngram_range=(1,2), lowercase=True)
        X = vec.fit_transform(texts)
        sim = cosine_similarity(X)
        print("[REP] TF-IDF            | [SIM] cosine  | semantic=DATASET_SPECIFIC")
        return X, sim, "tfidf", "cosine"

    elif rep_kind == "lsa":
        # Dataset specific. Factors learned from your corpus only.
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import normalize
        from sklearn.metrics.pairwise import cosine_similarity
        vec = TfidfVectorizer(min_df=2, ngram_range=(1,2), lowercase=True)
        X_tfidf = vec.fit_transform(texts)
        svd = TruncatedSVD(n_components=300, random_state=123)
        Z = svd.fit_transform(X_tfidf)
        Z = normalize(Z, norm="l2")
        sim = cosine_similarity(Z)
        print("[REP] LSA(on TF-IDF)    | [SIM] cosine  | semantic=DATASET_SPECIFIC")
        return Z, sim, "lsa(300 on tfidf)", "cosine"

    elif rep_kind == "jaccard":
        # Lexical overlap. Not semantic. Works on sparse without pairwise_distances.
        from sklearn.feature_extraction.text import CountVectorizer
        vec = CountVectorizer(min_df=1, ngram_range=(1,1), lowercase=True, binary=True)
        B = vec.fit_transform(texts).tocsr()  # [N x V] sparse binary
        N = B.shape[0]

        # |row| for each doc
        row_sums = np.asarray(B.sum(axis=1)).ravel().astype(np.float32)

        # Build dense similarity in blocks to keep memory reasonable
        sim = np.empty((N, N), dtype=np.float32)
        block = 256  # you can raise to 512 or 1024 if you have more RAM

        for i0 in range(0, N, block):
            i1 = min(i0 + block, N)
            # intersections = A[i] * A[j]^T since entries are 0 or 1
            inter = B[i0:i1].dot(B.T).toarray().astype(np.float32)  # [block x N]
            unions = row_sums[i0:i1][:, None] + row_sums[None, :] - inter
            # Jaccard = |A∩B| / |A∪B|, define 0 when union is 0
            with np.errstate(divide="ignore", invalid="ignore"):
                sim_block = np.where(unions > 0, inter / unions, 0.0)
            sim[i0:i1, :] = sim_block

        print("[REP] Binary BoW        | [SIM] jaccard | semantic=LEXICAL_ONLY")
        return B, sim, "binary-bow", "jaccard"

    else:
        raise ValueError(f"Unknown representation: {rep_kind}")


# ---------- Main analysis ----------
def main():
    # Outdir with label level suffix
    outdir = f"{OUTDIR_BASE}_{LABEL_LEVEL}"
    os.makedirs(outdir, exist_ok=True)

    messages = load_json(ANNOTATION_FILE)
    mapping  = load_json(MAPPING_FILE)

    texts, labels = [], []
    for msg in messages:
        for ann in msg.get("annotations", []):
            t = get_ann_text(ann)
            if not t:
                continue
            if LABEL_LEVEL in ("code", "subcode"):
                y = choose_label_by_level(ann, LABEL_LEVEL, mapping)
            elif LABEL_LEVEL == "combined":
                lv = extract_label_levels(ann, mapping)
                y = make_combined_label(lv["code"], lv["subcode"], require_both=REQUIRE_BOTH_FOR_COMBINED)
            else:
                raise ValueError(f"Unknown LABEL_LEVEL: {LABEL_LEVEL}")
            if y:
                texts.append(t)
                labels.append(y)

    N = len(texts)
    unique_labels = sorted(set(labels))
    print(f"[INFO] Clauses with usable text+label: {N}")
    print(f"[INFO] Unique labels: {len(unique_labels)}")

    # Optional legend for combined labels
    if LABEL_LEVEL == "combined":
        legend_path = os.path.join(outdir, "combined_label_legend.txt")
        with open(legend_path, "w", encoding="utf-8") as f:
            f.write("Format: CODE||SUBCODE\n")
            for k in sorted(set(labels)):
                f.write(k + "\n")

    # Representation and similarity
    X, sim, rep_name, sim_name = build_repr_and_sim(texts, REP_KIND, sbert_model=SBERT_MODEL)
    np.fill_diagonal(sim, -1.0)
    print(f"[INFO] Representation: {rep_name} | Similarity: {sim_name}")

    # Top-k neighbors per item
    topk_idx = np.argpartition(sim, -TOP_K, axis=1)[:, -TOP_K:]
    row_sorted = np.take_along_axis(sim, topk_idx, axis=1)
    order = np.argsort(-row_sorted, axis=1)
    topk_idx_sorted = np.take_along_axis(topk_idx, order, axis=1)
    topk_sim_sorted = np.take_along_axis(row_sorted, order, axis=1)

    # Counters
    conflict_pairs = Counter()
    pair_sim_sums = defaultdict(float)
    per_item_cross = []
    examples = []
    label_counts = Counter(labels)

    for i in range(N):
        yi = labels[i]
        sims = topk_sim_sorted[i]
        idxs = topk_idx_sorted[i]
        cross, total_considered = 0, 0
        for s, j in zip(sims, idxs):
            if s < SIM_THRESHOLD:
                continue
            total_considered += 1
            yj = labels[j]
            if yj != yi:
                cross += 1
                a, b = sorted([yi, yj])
                conflict_pairs[(a, b)] += 1
                pair_sim_sums[(a, b)] += float(s)
                if len(examples) < MAX_EXAMPLE_ROWS:
                    examples.append([texts[i], yi, texts[j], yj, float(s)])
        rate = cross / total_considered if total_considered > 0 else 0.0
        per_item_cross.append(rate)

    mean_cross = float(np.mean(per_item_cross)) if per_item_cross else 0.0
    median_cross = float(np.median(per_item_cross)) if per_item_cross else 0.0
    print(f"[INFO] Mean cross-label neighbor rate (k={TOP_K}, thr={SIM_THRESHOLD}): {mean_cross:.4f}")
    print(f"[INFO] Median cross-label neighbor rate: {median_cross:.4f}")

    # Samples
    with open(os.path.join(outdir, "nn_conflict_samples.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["anchor_text", "anchor_label", "neighbor_text", "neighbor_label", "similarity"])
        w.writerows(examples)
    print("[OK] Saved nn_conflict_samples.csv")

    # Detailed pair table with label totals and ratios
    pairs_path = os.path.join(outdir, "ambiguous_label_pairs_detailed.csv")
    with open(pairs_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "label_A","label_B","count","avg_similarity",
            "n_A","n_B","ratio_A=count/n_A","ratio_B=count/n_B"
        ])
        for (a, b), cnt in conflict_pairs.most_common():
            nA, nB = label_counts[a], label_counts[b]
            avg_sim = pair_sim_sums[(a, b)] / cnt if cnt else 0.0
            rA = cnt / nA if nA else 0.0
            rB = cnt / nB if nB else 0.0
            w.writerow([a, b, cnt, round(avg_sim, 4), nA, nB, round(rA, 4), round(rB, 4)])
    print(f"[OK] Saved {os.path.basename(pairs_path)}")

    # Per label disagreement
    per_label_rates = defaultdict(list)
    for r, y in zip(per_item_cross, labels):
        per_label_rates[y].append(r)
    with open(os.path.join(outdir, "per_label_disagreement.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "avg_cross_label_rate", "n_items"])
        for lab in sorted(per_label_rates.keys()):
            vals = per_label_rates[lab]
            w.writerow([lab, float(np.mean(vals)), len(vals)])
    print("[OK] Saved per_label_disagreement.csv")

    # Label totals
    with open(os.path.join(outdir, "label_totals.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["label", "n_items"])
        for lab in sorted(label_counts.keys()):
            w.writerow([lab, label_counts[lab]])
    print("[OK] Saved label_totals.csv")

    # Clustering purity
    from sklearn.cluster import KMeans
    k = max(len(unique_labels), 2)
    km = KMeans(n_clusters=k, n_init=10, random_state=123)
    km.fit(X)  # works for dense or CSR
    cluster_ids = km.labels_
    cluster_to_labels = defaultdict(list)
    for cid, y in zip(cluster_ids, labels):
        cluster_to_labels[cid].append(y)
    majority_sum = 0
    for cid, lst in cluster_to_labels.items():
        majority_sum += Counter(lst).most_common(1)[0][1]
    purity = majority_sum / N if N > 0 else 0.0
    with open(os.path.join(outdir, "cluster_purity.json"), "w", encoding="utf-8") as f:
        json.dump({"k": k, "purity": purity, "N": N}, f, indent=2)
    print(f"[INFO] Cluster purity (k={k}): {purity:.4f}")
    print("[OK] Saved cluster_purity.json")

    # Summary
    summary = {
        "N": N,
        "num_labels": len(unique_labels),
        "mean_cross_label_neighbor_rate": mean_cross,
        "median_cross_label_neighbor_rate": median_cross,
        "top_k_neighbors": TOP_K,
        "similarity_threshold": SIM_THRESHOLD,
        "representation": REP_KIND,
        "embedding_name": rep_name,
        "similarity_metric": sim_name,
        "label_level": LABEL_LEVEL,
        "require_both_for_combined": REQUIRE_BOTH_FOR_COMBINED,
    }
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[OK] Saved summary.json")


if __name__ == "__main__":
    main()
