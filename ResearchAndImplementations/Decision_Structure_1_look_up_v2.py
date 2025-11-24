# phrase_debug_and_lookup.py
"""
Run:
    python phrase_debug_and_lookup.py

What it does:
  - Uses your spaCy splitter (Decision_Structure_1_chucking_v3) to get clauses and phrase chunks
  - Prints phrase details (text, char offsets, token indices, clause text)
  - Looks up each phrase against your phrase index (built from phrase_pool.csv + phrase_label_edges.csv)
  - Prints candidate keys, similarities, and per-label scores, then a sentence-level label distribution

Edit CONFIG below for:
  - CSV paths (phrase_pool.csv, phrase_label_edges.csv)
  - cache path for the built index
  - sample sentences to test
  - fuzzy thresholds, scoring weights, optional keyword-group boosts
"""

import os, csv, json, math, re
from collections import defaultdict, Counter
from typing import Dict, Any, List, Tuple, Optional, Iterable, Set

# ======== CONFIG ========
CONFIG = {
    # Required CSVs from your stats run
    "PHRASE_POOL_CSV": f"Phrase_out/out/-phrase_pool.csv",
    "EDGES_CSV": f"Phrase_out/out/-phrase_label_edges.csv",

    # Cache for the prebuilt phrase index
    "CACHE_JSON": f"Phrase_out/out/cache_phrase_index.json",

    # Sentences to test (ignored if you import and call debug_and_lookup(...) yourself)
    "SENTENCES": [
        "I will be down at Org3 for some other testing.",
        # "we ran out of time yesterday afternoon",
        # "SO sorry we ran out of time yesterday afternoon and Dr. Person1 had",
        # "The good doctor is swamped in a emails and she wants to make sure he sees it",
    ],

    # Matching thresholds
    "JACCARD_NGRAM": 3,
    "JACCARD_MIN": 0.85,      # primary fuzzy gate
    "TOKEN_RATIO_MIN": 0.85,  # secondary fuzzy gate

    # Scoring
    "PMI_FLOOR": 0.0,         # clamp negative PMI to this before scoring
    "COFREQ_LOG_BASE": 1.0,   # weight term uses log1p(cofreq * base)
    "LABEL_PRIOR_SCALE": 0.5, # add prior mass from label cofreq totals
    "SOFTMAX_T": 1.0,         # temperature for sentence-level label probabilities

    # Optional boosts: label -> list of substrings or regex
    "KEYWORD_GROUPS": {
        # "Scheduling": ["schedule", r"\bappointment\b", r"\breschedul"],
        # "Apology": ["sorry", "apolog"]
    },

    # How many candidate keys to print per phrase
    "MAX_MATCHES_PER_PHRASE": 5,
}
# ======== END CONFIG ========


# ======== import your spaCy v3 splitter ========
# The module must be on PYTHONPATH or in the same folder.
from Decision_Structure_1_chucking_v3 import nlp, get_clause_spans, get_phrase_chunks


# ======== text utils ========
def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    # placeholders for numbers/dates/times, so keys are more stable
    s = re.sub(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", "<date>", s)
    s = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "<date>", s)
    s = re.sub(r"\b\d{1,2}:\d{2}(?:\s*[ap]m)?\b", "<time>", s)
    s = re.sub(r"\b\d+(?:\.\d+)?\b", "<num>", s)
    return s

def ngrams(s: str, n: int) -> Set[str]:
    s2 = f" {s} "
    return {s2[i:i+n] for i in range(0, max(0, len(s2) - n + 1))}

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def token_ratio(a: str, b: str) -> float:
    at = a.split()
    bt = b.split()
    if not at or not bt:
        return 0.0
    A, B = set(at), set(bt)
    inter = len(A & B)
    union = len(A | B)
    return inter / union if union else 0.0

def softmax(scores: Dict[str, float], T: float = 1.0) -> Dict[str, float]:
    if not scores:
        return {}
    if T <= 0:
        T = 1.0
    m = max(scores.values())
    exps = {k: math.exp((v - m) / T) for k, v in scores.items()}
    Z = sum(exps.values())
    return {k: exps[k] / Z for k in scores} if Z > 0 else {k: 0.0 for k in scores}

def hits_keyword_group(text: str, patterns: List[str]) -> bool:
    t = text.lower()
    for p in patterns:
        # allow regex or substring
        if p.startswith("^") or p.endswith("$") or ("\\" in p) or ("[" in p and "]" in p):
            if re.search(p, t, flags=re.IGNORECASE):
                return True
        if p.lower() in t:
            return True
    return False


# ======== index building and loading ========
def _as_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default

def _as_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        s = str(x)
        if s == "" or s.lower() in ("none", "nan"):
            return default
        return float(s)
    except Exception:
        return default

def load_phrase_pool(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "phrase_text": r.get("phrase_text", ""),
                "phrase_type": r.get("phrase_type", "OTHER"),
                "occurrences": _as_int(r.get("occurrences", 0)),
                "doc_freq": _as_int(r.get("doc_freq", 0)),
                "top_label_by_PMI": r.get("top_label_by_PMI", ""),
                "top_label_PMI": _as_float(r.get("top_label_PMI", None), None),
                "label_purity": _as_float(r.get("label_purity", 0.0), 0.0),
                "label_entropy": _as_float(r.get("label_entropy", 0.0), 0.0),
            })
    return rows

def load_edges(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "phrase_text": r.get("phrase_text", ""),
                "phrase_type": r.get("phrase_type", "OTHER"),
                "label": r.get("label", ""),
                "doc_cofreq": _as_int(r.get("doc_cofreq", 0)),
                "pmi": _as_float(r.get("pmi", None), None),
            })
    return rows

def build_index(pool_rows: List[Dict[str, Any]], edge_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    key_to_scores = defaultdict(lambda: defaultdict(lambda: {"cofreq": 0, "pmi": None}))
    phrase_info = {}
    label_prior = Counter()

    # label prior from edge totals
    for e in edge_rows:
        label_prior[e["label"]] += e["doc_cofreq"]

    for e in edge_rows:
        key = (normalize_text(e["phrase_text"]), e["phrase_type"])
        cur = key_to_scores[key][e["label"]]
        cur["cofreq"] += e["doc_cofreq"]
        if e["pmi"] is not None:
            cur["pmi"] = max(cur["pmi"], e["pmi"]) if cur["pmi"] is not None else e["pmi"]

    for r in pool_rows:
        key = (normalize_text(r["phrase_text"]), r["phrase_type"])
        info = phrase_info.get(key, {
            "orig_examples": set(),
            "doc_freq": 0,
            "occurrences": 0,
            "top_label_by_PMI": "",
            "top_label_PMI": 0.0,
            "label_purity": 0.0,
            "label_entropy": 0.0,
        })
        info["orig_examples"].add(r["phrase_text"])
        info["doc_freq"] = max(info["doc_freq"], r["doc_freq"])
        info["occurrences"] += r["occurrences"]
        if r["top_label_PMI"] is not None and r["top_label_PMI"] > info["top_label_PMI"]:
            info["top_label_by_PMI"] = r["top_label_by_PMI"]
            info["top_label_PMI"] = r["top_label_PMI"]
        info["label_purity"] = max(info["label_purity"], r["label_purity"])
        info["label_entropy"] = max(info["label_entropy"], r["label_entropy"])
        phrase_info[key] = info

    # convert sets to lists
    for k in phrase_info:
        phrase_info[k]["orig_examples"] = list(sorted(phrase_info[k]["orig_examples"]))[:5]

    # JSON keys
    return {
        "key_to_scores": {f"{k[0]}||{k[1]}": v for k, v in key_to_scores.items()},
        "phrase_info": {f"{k[0]}||{k[1]}": v for k, v in phrase_info.items()},
        "label_prior": dict(label_prior)
    }

def save_cache(index: Dict[str, Any], path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

def load_cache(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_or_load_index(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cache = load_cache(cfg["CACHE_JSON"])
    if cache:
        return cache
    pool = load_phrase_pool(cfg["PHRASE_POOL_CSV"])
    edges = load_edges(cfg["EDGES_CSV"])
    index = build_index(pool, edges)
    save_cache(index, cfg["CACHE_JSON"])
    return index


# ======== mapping helpers for printing offsets ========
def char_to_token_span(doc, char_start: int, char_end: int) -> Tuple[int, int]:
    """Convert char offsets in doc to token indices [start, end)."""
    t_start = None
    for t in doc:
        if t.idx + len(t) > char_start:
            t_start = t.i
            break
    if t_start is None:
        t_start = len(doc) - 1
    t_end = t_start
    for t in doc[t_start:]:
        if t.idx + len(t) >= char_end:
            t_end = t.i + 1
            break
    else:
        t_end = len(doc)
    return t_start, t_end


# ======== lookup ========
def find_candidates_for_phrase(text: str, ptype: str, index: Dict[str, Any], cfg: Dict[str, Any]) -> List[Tuple[str, float]]:
    """Return candidate keys 'norm||ptype' with similarity score."""
    norm = normalize_text(text)
    key_exact = f"{norm}||{ptype}"
    all_keys = list(index["phrase_info"].keys())
    out = []
    if key_exact in index["phrase_info"]:
        out.append((key_exact, 1.0))
    n = cfg["JACCARD_NGRAM"]
    a = ngrams(norm, n)
    for k in all_keys:
        k_norm, k_type = k.split("||", 1)
        if k_type != ptype or k == key_exact:
            continue
        sim = jaccard(a, ngrams(k_norm, n))
        if sim >= cfg["JACCARD_MIN"]:
            if token_ratio(norm, k_norm) >= cfg["TOKEN_RATIO_MIN"]:
                out.append((k, sim))
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:cfg["MAX_MATCHES_PER_PHRASE"]]

def score_match_labels(key: str, phrase_text: str, index: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compute per-label scores for a matched key."""
    lbls = []
    key_scores = index["key_to_scores"].get(key, {})
    for L, stats in key_scores.items():
        pmi = stats["pmi"]
        cofreq = stats["cofreq"]
        if pmi is None:
            continue
        pmi_clamped = max(pmi, cfg["PMI_FLOOR"])
        s = pmi_clamped * math.log1p(cofreq * cfg["COFREQ_LOG_BASE"])
        patterns = cfg["KEYWORD_GROUPS"].get(L, [])
        if patterns and hits_keyword_group(phrase_text, patterns):
            s *= 1.0 * CONFIG["GROUP_BOOST"]  # boost
        if s > 0:
            lbls.append({"label": L, "score": s, "cofreq": cofreq, "pmi": pmi})
    lbls.sort(key=lambda d: d["score"], reverse=True)
    return lbls


# ======== main routine: split + print + lookup ========
def debug_and_lookup(sentence: str, cfg: Dict[str, Any] = CONFIG) -> Dict[str, Any]:
    index = build_or_load_index(cfg)
    doc = nlp(sentence)

    print("=" * 88)
    print("INPUT:", sentence)
    sent_label_scores = Counter()
    per_phrase_outputs = []

    for si, sent in enumerate(doc.sents, 1):
        print(f"\nSentence {si}: {sent.text}")
        clauses = get_clause_spans(sent)
        for ci, cl in enumerate(clauses, 1):
            print(f"  Clause {ci}: {cl.text}")
            phrases = get_phrase_chunks(cl)

            cursor = 0
            for p in phrases:
                ptxt = p["text"]
                ptype = p.get("type", "OTHER")
                rel = cl.text.find(ptxt, cursor)
                if rel < 0:
                    # whitespace tolerant fallback
                    squeezed_clause = re.sub(r"\s+", " ", cl.text)
                    squeezed_ptxt = re.sub(r"\s+", " ", ptxt)
                    rel = squeezed_clause.find(squeezed_ptxt, 0)
                    if rel < 0:
                        # could not align offsets, still print text and type
                        print(f"    [{ptype}] {ptxt}")
                        candidates = find_candidates_for_phrase(ptxt, ptype, index, cfg)
                        matches_out = []
                        for key, sim in candidates:
                            lbls = score_match_labels(key, ptxt, index, cfg)
                            examples = index["phrase_info"].get(key, {}).get("orig_examples", [])
                            print(f"      ~{sim:.2f}  {key}  " + ", ".join(f"{d['label']}:{d['score']:.2f}" for d in lbls[:3]))
                            matches_out.append({"key": key, "sim": sim, "labels": lbls, "examples": examples})
                            for d in lbls:
                                sent_label_scores[d["label"]] += d["score"] * sim
                        per_phrase_outputs.append({"text": ptxt, "type": ptype, "matches": matches_out})
                        continue
                    # Map rel from squeezed to original roughly
                    nonspace = 0
                    target = len(squeezed_clause[:rel].replace(" ", ""))
                    orig_rel = 0
                    while orig_rel < len(cl.text) and nonspace < target:
                        if not cl.text[orig_rel].isspace():
                            nonspace += 1
                        orig_rel += 1
                    rel = orig_rel

                # absolute char offsets in the whole doc
                char_start = cl.start_char + rel
                char_end = char_start + len(ptxt)
                t_start, t_end = char_to_token_span(doc, char_start, char_end)

                print(f"    [{ptype}] {ptxt}  | chars {char_start}..{char_end} | toks {t_start}..{t_end}")

                # lookup
                candidates = find_candidates_for_phrase(ptxt, ptype, index, cfg)
                matches_out = []
                if not candidates:
                    print(f"      (no index match)")
                for key, sim in candidates:
                    lbls = score_match_labels(key, ptxt, index, cfg)
                    examples = index["phrase_info"].get(key, {}).get("orig_examples", [])
                    # print top labels
                    if lbls:
                        lbl_str = ", ".join(f"{d['label']}:{d['score']:.2f}" for d in lbls[:3])
                    else:
                        lbl_str = "(no positive label score)"
                    print(f"      ~{sim:.2f}  {key}  {lbl_str}")
                    if examples:
                        print(f"         ex: {examples[:3]}")
                    matches_out.append({"key": key, "sim": sim, "labels": lbls, "examples": examples})
                    for d in lbls:
                        sent_label_scores[d["label"]] += d["score"] * sim

                per_phrase_outputs.append({
                    "text": ptxt, "type": ptype,
                    "char_span": [char_start, char_end],
                    "tok_span": [t_start, t_end],
                    "matches": matches_out
                })

    # add a small prior
    for L, prior_cnt in index.get("label_prior", {}).items():
        if prior_cnt > 0:
            sent_label_scores[L] += cfg["LABEL_PRIOR_SCALE"] * math.log1p(prior_cnt)

    probs = softmax(dict(sent_label_scores), T=cfg["SOFTMAX_T"])

    # summary
    if sent_label_scores:
        print("\nLabel scores")
        for L, s in sorted(sent_label_scores.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {L:20s}  {s:.3f}   P={probs.get(L,0):.3f}")
    else:
        print("\n(no label signal found)")

    return {"label_scores": dict(sent_label_scores), "label_probs": probs, "per_phrase": per_phrase_outputs}


if __name__ == "__main__":
    cfg = CONFIG
    # build or load the index once
    _ = build_or_load_index(cfg)
    for s in cfg["SENTENCES"]:
        debug_and_lookup(s, cfg)
