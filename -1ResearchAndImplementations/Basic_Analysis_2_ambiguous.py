"""
Multi-representation ambiguity analysis (fixed label level).

Pick LABEL_LEVEL once, then loop representations:
- sbert, tfidf, lsa, jaccard

Outputs (under /mnt/data/ambiguity_compare_<LABEL_LEVEL>/):
- <rep>/nn_conflict_samples.csv
- <rep>/ambiguous_label_pairs_detailed.csv
- <rep>/per_label_disagreement.csv
- merged_all_conflict_examples.csv
- merged_all_label_pairs.csv
- merged_all_per_label_disagreement.csv
- summary_by_method.json
- optional figures:
    merged_top_ambiguous_pairs.png
    merged_per_label_disagreement.png
"""

from __future__ import annotations
import json, math, csv, sys, warnings
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np

# plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

# ------------- CONFIG -------------
# BASE = Path("Bethesda_output")
# INPUT_JSON = BASE / "Bethesda_processed_messages_with_annotations.json"

BASE = Path("EPPC_output_json/CleanedData")
INPUT_JSON = BASE / "processed_messages_with_annotations.json"


# choose one label level to compare across methods
# LABEL_LEVEL = "subcode"   # "code" | "subcode" | "combined"
# LABEL_LEVEL = "code"   # "code" | "subcode" | "combined"
LABEL_LEVEL = "combined"   # "code" | "subcode" | "combined"


REPRESENTATIONS = ["sbert", "tfidf", "lsa", "jaccard"]

K_NEIGHBORS = 10
SIM_THRESHOLD = 0.60
LSA_DIM = 200
MAX_EXAMPLES_PER_PAIR = 25
MAKE_FIGURES = True

OUT_ROOT = BASE / f"ambiguity_compare_{LABEL_LEVEL}"
OUT_ROOT.mkdir(exist_ok=True)

# ------------- IO helpers -------------
def save_csv(rows: List[Dict], path: Path, header: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

# ------------- data loading -------------
@dataclass
class Span:
    text: str
    label: str
    message_id: int
    text_id: str

def load_spans(input_json: Path, level: str) -> List[Span]:
    data = json.loads(input_json.read_text(encoding="utf-8"))
    spans: List[Span] = []
    for rec in data:
        mid = rec.get("message_id")
        for ann in rec.get("annotations", []):
            labels = ann.get("code", [])
            if not labels: continue
            if level == "code":
                lbl = labels[0]
            elif level == "subcode":
                if len(labels) < 2: 
                    continue
                lbl = labels[1]
            elif level == "combined":
                lbl = f"{labels[0]} → {labels[1]}" if len(labels) >= 2 else labels[0]
            else:
                raise ValueError("LABEL_LEVEL must be code | subcode | combined")
            txt = (ann.get("text") or "").strip()
            if txt:
                spans.append(Span(txt, lbl, mid, ann.get("text_id","")))
    return spans

# ------------- representations -------------
def build_sbert(texts: List[str]) -> np.ndarray | None:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        warnings.warn("sentence-transformers not available. Falling back to TF-IDF in SBERT branch.")
        return None
    name = "all-MiniLM-L6-v2"
    try:
        model = SentenceTransformer(name)
        emb = np.array(model.encode(texts, show_progress_bar=False, normalize_embeddings=True))
        return emb
    except Exception as e:
        warnings.warn(f"SBERT load failed: {e}; fallback to TF-IDF.")
        return None

def build_tfidf(texts: List[str]):
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1,2), min_df=2, max_df=0.9)
    X = vec.fit_transform(texts)
    return X

def build_lsa(texts: List[str], k: int):
    X = build_tfidf(texts)
    k = max(1, min(k, X.shape[1]-1)) if X.shape[1] > 1 else 1
    svd = TruncatedSVD(n_components=k, random_state=42)
    Z = svd.fit_transform(X)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    return Z

def jaccard_sets(texts: List[str]) -> List[set]:
    tok = lambda t: set(t.lower().split())
    return [tok(t) for t in texts]

def jaccard_sim_matrix(sets: List[set]) -> np.ndarray:
    n = len(sets); S = np.zeros((n,n), dtype=float)
    for i in range(n):
        a = sets[i]
        for j in range(i, n):
            b = sets[j]
            inter = len(a & b)
            uni = len(a | b) or 1
            s = inter / uni
            S[i,j] = S[j,i] = s
    return S

# ------------- neighbors -------------
def cosine_neighbors(A, k: int):
    # A can be dense array; metric cosine -> distance = 1 - sim
    nn = NearestNeighbors(n_neighbors=min(k+1, A.shape[0]), metric="cosine", n_jobs=-1)
    nn.fit(A)
    dist, idx = nn.kneighbors(A, return_distance=True)
    return dist[:,1:], idx[:,1:]

def tfidf_neighbors(X, k: int):
    nn = NearestNeighbors(n_neighbors=min(k+1, X.shape[0]), metric="cosine", n_jobs=-1)
    nn.fit(X)
    dist, idx = nn.kneighbors(X, return_distance=True)
    return dist[:,1:], idx[:,1:]

def jaccard_neighbors(S: np.ndarray, k: int):
    D = 1.0 - S
    idx = np.argsort(D, axis=1)[:, 1:k+1]
    rows = np.arange(D.shape[0])[:, None]
    dist = D[rows, idx]
    return dist, idx

def sim_from_dist(dist, metric: str):
    return 1.0 - dist  # both cosine and jaccard here

# ------------- per method run -------------
def run_one_method(spans: List[Span], rep: str, out_dir: Path) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = [s.text for s in spans]
    labels = [s.label for s in spans]
    uniq = sorted(set(labels))

    # features + neighbors
    metric = "cosine"
    if rep == "sbert":
        emb = build_sbert(texts)
        if emb is None:
            X = build_tfidf(texts)
            dist, idx = tfidf_neighbors(X, K_NEIGHBORS)
        else:
            dist, idx = cosine_neighbors(emb, K_NEIGHBORS)
    elif rep == "tfidf":
        X = build_tfidf(texts)
        dist, idx = tfidf_neighbors(X, K_NEIGHBORS)
    elif rep == "lsa":
        Z = build_lsa(texts, LSA_DIM)
        dist, idx = cosine_neighbors(Z, K_NEIGHBORS)
    elif rep == "jaccard":
        sets = jaccard_sets(texts)
        S = jaccard_sim_matrix(sets)
        dist, idx = jaccard_neighbors(S, K_NEIGHBORS)
        metric = "jaccard"
    else:
        raise ValueError("rep must be sbert | tfidf | lsa | jaccard")

    sims = sim_from_dist(dist, metric)

    # collect conflicts and stats
    pair_counter = Counter()
    pair_sims = defaultdict(list)
    per_label_disagree = Counter()
    per_label_total = Counter()
    examples = []

    N = len(spans)
    for i in range(N):
        src_lab = labels[i]
        src_text = texts[i]
        per_label_total[src_lab] += sims.shape[1]
        for k in range(sims.shape[1]):
            j = idx[i, k]
            sim = sims[i, k]
            if sim < SIM_THRESHOLD: 
                continue
            tgt_lab = labels[j]
            if tgt_lab != src_lab:
                a, b = sorted([src_lab, tgt_lab])
                key = (a, b)
                pair_counter[key] += 1
                pair_sims[key].append(float(sim))
                per_label_disagree[src_lab] += 1
                # cap per pair
                if sum(1 for r in examples if r["pair"] == f"{a} || {b}") < MAX_EXAMPLES_PER_PAIR:
                    examples.append({
                        "pair": f"{a} || {b}",
                        "src_label": src_lab, "src_text": src_text,
                        "nbr_label": tgt_lab, "nbr_text": texts[j],
                        "similarity": round(float(sim), 4)
                    })

    # write per method outputs
    save_csv(examples, out_dir / "nn_conflict_samples.csv",
             ["pair","src_label","src_text","nbr_label","nbr_text","similarity"])

    rows_pairs = []
    for (a, b), cnt in sorted(pair_counter.items(), key=lambda kv: -kv[1]):
        sims_list = pair_sims[(a, b)]
        rows_pairs.append({
            "label_a": a, "label_b": b,
            "conflict_count": cnt,
            "mean_similarity": round(float(np.mean(sims_list)), 4),
            "median_similarity": round(float(np.median(sims_list)), 4),
            "max_similarity": round(float(np.max(sims_list)), 4),
            "min_similarity": round(float(np.min(sims_list)), 4)
        })
    save_csv(rows_pairs, out_dir / "ambiguous_label_pairs_detailed.csv",
             ["label_a","label_b","conflict_count","mean_similarity","median_similarity","max_similarity","min_similarity"])

    # per label disagreement
    per_lab_rows = []
    for lab in sorted(set(labels), key=lambda x: (-per_label_disagree[x], x)):
        tot = per_label_total.get(lab, 0) or 1
        rate = per_label_disagree[lab] / tot
        per_lab_rows.append({
            "label": lab,
            "neighbor_disagreement": per_label_disagree[lab],
            "neighbor_total": tot,
            "disagreement_rate": round(float(rate), 4)
        })
    save_csv(per_lab_rows, out_dir / "per_label_disagreement.csv",
             ["label","neighbor_disagreement","neighbor_total","disagreement_rate"])

    # summary for this method
    summary = {
        "rep": rep,
        "num_spans": N,
        "num_labels": len(uniq),
        "num_conflict_pairs": int(sum(pair_counter.values())),
        "num_distinct_label_pairs": int(len(pair_counter)),
        "avg_disagreement_rate": round(float(np.mean([r["disagreement_rate"] for r in per_lab_rows])) if per_lab_rows else 0.0, 4),
        "median_disagreement_rate": round(float(np.median([r["disagreement_rate"] for r in per_lab_rows])) if per_lab_rows else 0.0, 4)
    }
    save_json(summary, out_dir / "summary.json")
    return {
        "pairs": rows_pairs,
        "examples": examples,
        "per_label": per_lab_rows,
        "summary": summary
    }

# ------------- aggregation across methods -------------
def aggregate_across_methods(results_by_rep: Dict[str, Dict]):
    # merge examples
    all_examples = []
    for rep, res in results_by_rep.items():
        for r in res["examples"]:
            rr = dict(r); rr["method"] = rep
            all_examples.append(rr)
    if all_examples:
        save_csv(all_examples, OUT_ROOT / "merged_all_conflict_examples.csv",
                 ["method","pair","src_label","src_text","nbr_label","nbr_text","similarity"])

    # merge pairs
    all_pairs = []
    for rep, res in results_by_rep.items():
        for r in res["pairs"]:
            rr = dict(r); rr["method"] = rep
            all_pairs.append(rr)
    if all_pairs:
        save_csv(all_pairs, OUT_ROOT / "merged_all_label_pairs.csv",
                 ["method","label_a","label_b","conflict_count","mean_similarity","median_similarity","max_similarity","min_similarity"])

    # merge per label disagreement
    all_per = []
    for rep, res in results_by_rep.items():
        for r in res["per_label"]:
            rr = dict(r); rr["method"] = rep
            all_per.append(rr)
    if all_per:
        save_csv(all_per, OUT_ROOT / "merged_all_per_label_disagreement.csv",
                 ["method","label","neighbor_disagreement","neighbor_total","disagreement_rate"])

    # merged summary
    by_method = {rep: res["summary"] for rep, res in results_by_rep.items()}
    save_json(by_method, OUT_ROOT / "summary_by_method.json")

    # figures
    if MAKE_FIGURES and all_pairs:
        # top ambiguous pairs across all methods
        top_pairs = sorted(all_pairs, key=lambda r: -r["conflict_count"])[:25]
        labels_xy = [f'{r["label_a"]} | {r["label_b"]} ({r["method"]})' for r in top_pairs]
        counts = [r["conflict_count"] for r in top_pairs]
        plt.figure(figsize=(11, 7))
        plt.barh(range(len(labels_xy)), counts)
        plt.yticks(range(len(labels_xy)), labels_xy, fontsize=8)
        plt.gca().invert_yaxis()
        plt.title(f"Top ambiguous pairs across methods [{LABEL_LEVEL}]")
        plt.tight_layout()
        plt.savefig(OUT_ROOT / "merged_top_ambiguous_pairs.png", dpi=220)
        plt.close()

    if MAKE_FIGURES and all_per:
        # per label disagreement: show max rate per label across methods
        # collapse to label -> max rate
        best = {}
        for r in all_per:
            lab = r["label"]; rate = float(r["disagreement_rate"])
            best[lab] = max(best.get(lab, 0.0), rate)
        items = sorted(best.items(), key=lambda kv: -kv[1])[:30]
        labs = [k for k, _ in items]; rates = [v for _, v in items]
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(labs)), rates)
        plt.yticks(range(len(labs)), labs, fontsize=8)
        plt.gca().invert_yaxis()
        plt.xlabel("Max disagreement rate across methods")
        plt.title(f"Per label disagreement (max across methods) [{LABEL_LEVEL}]")
        plt.tight_layout()
        plt.savefig(OUT_ROOT / "merged_per_label_disagreement.png", dpi=220)
        plt.close()

# ------------- main -------------
def main():
    print(f"[load] {INPUT_JSON}")
    spans = load_spans(INPUT_JSON, LABEL_LEVEL)
    if not spans:
        print("No spans found. Check LABEL_LEVEL or input file.")
        sys.exit(1)

    print(f"[info] N spans: {len(spans)} | Label level: {LABEL_LEVEL}")
    results_by_rep = {}
    for rep in REPRESENTATIONS:
        print(f"=== {LABEL_LEVEL} / {rep} ===")
        out_dir = OUT_ROOT / rep
        res = run_one_method(spans, rep, out_dir)
        results_by_rep[rep] = res

    print("[aggregate] merging outputs across methods")
    aggregate_across_methods(results_by_rep)
    print(f"[done] Outputs written under: {OUT_ROOT}")

if __name__ == "__main__":
    main()
