# phrase_lookup_config.py
# Run:
#   python phrase_lookup_config.py
# Or import and call score_sentence(["we ran out of time", "yesterday afternoon"])

import csv, json, os, math, re
from collections import defaultdict, Counter
from typing import Dict, Any, List, Tuple, Optional, Iterable, Set

# ========= CONFIG =========
CONFIG = {
    # Required inputs from your stats script
    "PHRASE_POOL_CSV": f"Phrase_out/out/-phrase_pool.csv",
    "EDGES_CSV": f"Phrase_out/out/-phrase_label_edges.csv",   # columns: phrase_text, phrase_type, label, doc_cofreq, pmi

    # Optional
    "TYPE_BY_LABEL_CSV": r"",                           # optional matrix to inspect later

    # Output cache for the index so next run is instant
    "CACHE_JSON": f"Phrase_out/out/phrase_index_cache.json",

    # Phrase fields to use when you only have the raw sentence
    # If you want to pass already chunked phrases, leave this and call score_sentence directly.
    "USE_INTERNAL_CHUNKER": False,                      # lightweight regex chunker if True

    # Normalization and fuzzy matching thresholds
    "JACCARD_NGRAM": 3,
    "JACCARD_MIN": 0.85,                                # fuzzy match threshold
    "TOKEN_RATIO_MIN": 0.85,                            # another safeguard

    # Scoring weights
    "PMI_FLOOR": 0.0,                                   # we clamp negative PMI to this
    "COFREQ_LOG_BASE": 1.0,                             # weight uses log1p(cofreq * base)
    "LABEL_PRIOR_SCALE": 0.5,                           # amount of prior mass from label docfreq
    "GROUP_BOOST": 1.5,                                 # multiplier if phrase hits a curated group keyword

    # Curated keyword groups: label -> list of substrings or regex (case insensitive)
    "KEYWORD_GROUPS": {
        # "Scheduling": ["schedule", r"\bappointment\b", r"\breschedule\b"],
        # "Billing": ["bill", "copay", "invoice"],
    },

    # Softmax temperature for probability output
    "SOFTMAX_T": 1.0
}
# ========= END CONFIG =========


# ---------- util ----------
def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    # replace dates and numbers with placeholders
    s = re.sub(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", "<date>", s)
    s = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "<date>", s)
    s = re.sub(r"\b\d{1,2}:\d{2}(?:\s*[ap]m)?\b", "<time>", s)
    s = re.sub(r"\b\d+(?:\.\d+)?\b", "<num>", s)
    return s

def ngrams(s: str, n: int) -> Set[str]:
    s = f" {s} "
    return {s[i:i+n] for i in range(0, max(0, len(s) - n + 1))}

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
    inter = len(set(at) & set(bt))
    union = len(set(at) | set(bt))
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
        if p.startswith("^") or p.endswith("$") or ("\\" in p) or ("[" in p and "]" in p):
            if re.search(p, t, flags=re.IGNORECASE):
                return True
        if p.lower() in t:
            return True
    return False


# ---------- index building ----------
def load_phrase_pool(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # keep only the columns we need
            rows.append({
                "phrase_text": r.get("phrase_text", ""),
                "phrase_type": r.get("phrase_type", "OTHER"),
                "occurrences": int(float(r.get("occurrences", "0") or 0)),
                "doc_freq": int(float(r.get("doc_freq", "0") or 0)),
                "top_label_by_PMI": r.get("top_label_by_PMI", ""),
                "top_label_PMI": float(r.get("top_label_PMI", "0") or 0),
                "label_purity": float(r.get("label_purity", "0") or 0),
                "label_entropy": float(r.get("label_entropy", "0") or 0),
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
                "doc_cofreq": int(float(r.get("doc_cofreq", "0") or 0)),
                "pmi": None if r.get("pmi", "") in ("", "None", "nan") else float(r["pmi"]),
            })
    return rows

def build_index(pool_rows: List[Dict[str, Any]], edge_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    # group scores per phrase key
    # key = (norm_text, phrase_type)
    phrase_info = {}
    label_prior = Counter()
    for e in edge_rows:
        label_prior[e["label"]] += e["doc_cofreq"]

    key_to_scores = defaultdict(lambda: defaultdict(lambda: {"cofreq": 0, "pmi": None}))
    for e in edge_rows:
        key = (normalize_text(e["phrase_text"]), e["phrase_type"])
        cur = key_to_scores[key][e["label"]]
        cur["cofreq"] += e["doc_cofreq"]
        # keep the best PMI for this pair
        if e["pmi"] is not None:
            cur["pmi"] = max(cur["pmi"], e["pmi"]) if cur["pmi"] is not None else e["pmi"]

    # attach meta from pool
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

    return {
        "key_to_scores": {f"{k[0]}||{k[1]}": v for k, v in key_to_scores.items()},
        "phrase_info": {f"{k[0]}||{k[1]}": v for k, v in phrase_info.items()},
        "label_prior": dict(label_prior)
    }

def save_cache(index: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

def load_cache(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------- lookup and scoring ----------
def find_candidates_for_phrase(text: str, ptype: str, index: Dict[str, Any], cfg: Dict[str, Any]) -> List[Tuple[str, float]]:
    """
    Returns a list of candidate keys "norm||ptype" with similarity score in [0,1].
    Includes exact and fuzzy matches.
    """
    norm = normalize_text(text)
    key_exact = f"{norm}||{ptype}"
    all_keys = list(index["phrase_info"].keys())
    # exact
    out = []
    if key_exact in index["phrase_info"]:
        out.append((key_exact, 1.0))

    # fuzzy on same type first
    n = cfg["JACCARD_NGRAM"]
    a = ngrams(norm, n)
    for k in all_keys:
        k_norm, k_type = k.split("||", 1)
        if k_type != ptype:
            continue
        if k == key_exact:
            continue
        sim = jaccard(a, ngrams(k_norm, n))
        if sim >= cfg["JACCARD_MIN"]:
            # secondary check to avoid false positives
            if token_ratio(norm, k_norm) >= cfg["TOKEN_RATIO_MIN"]:
                out.append((k, sim))
    # sort by similarity descending
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:20]

def score_labels_for_phrases(phrases: List[Tuple[str, str]],
                             index: Dict[str, Any],
                             cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    phrases: list of (phrase_text, phrase_type)
    Returns:
      {
        "label_scores": {label: score},
        "label_probs": {label: prob},
        "per_phrase": [
            {"phrase": "...", "type": "NP", "matches": [
                 {"key":"...", "sim": 0.92, "labels":[{"label":"L","score":...}, ...], "examples":[...]}
            ]}
          ]
      }
    """
    label_scores = Counter()
    per_phrase = []
    key_to_scores = index["key_to_scores"]
    phrase_info = index["phrase_info"]

    for text, ptype in phrases:
        cands = find_candidates_for_phrase(text, ptype, index, cfg)
        match_entries = []
        for key, sim in cands:
            lbls = []
            for L, stats in key_to_scores.get(key, {}).items():
                pmi = stats["pmi"]
                cofreq = stats["cofreq"]
                if pmi is None:
                    continue
                pmi_clamped = max(pmi, cfg["PMI_FLOOR"])
                s = pmi_clamped * math.log1p(cofreq * cfg["COFREQ_LOG_BASE"])
                # group boost if this text hits curated group for L
                patterns = cfg["KEYWORD_GROUPS"].get(L, [])
                if patterns and hits_keyword_group(text, patterns):
                    s *= cfg["GROUP_BOOST"]
                if s > 0:
                    lbls.append({"label": L, "score": s})
                    label_scores[L] += s * sim
            if lbls:
                ex = phrase_info.get(key, {}).get("orig_examples", []) if key in phrase_info else []
                match_entries.append({"key": key, "sim": round(sim, 3), "labels": lbls, "examples": ex})
        per_phrase.append({"phrase": text, "type": ptype, "matches": match_entries})

    # add a small prior
    for L, prior_cnt in index.get("label_prior", {}).items():
        if prior_cnt > 0:
            label_scores[L] += cfg["LABEL_PRIOR_SCALE"] * math.log1p(prior_cnt)

    probs = softmax(dict(label_scores), T=cfg["SOFTMAX_T"])
    return {"label_scores": dict(label_scores), "label_probs": probs, "per_phrase": per_phrase}


# ---------- optional tiny chunker ----------
# You can set USE_INTERNAL_CHUNKER=True as a quick test when you do not have your own splitter handy.
def tiny_chunk(sentence: str) -> List[Tuple[str, str]]:
    """
    A very rough fallback that splits on comma and 'and', then picks phrase types by heuristic.
    Replace with your spaCy or LLM chunks in production.
    """
    parts = re.split(r"[,;]| and ", sentence)
    out = []
    for p in parts:
        t = p.strip()
        if not t:
            continue
        # naive type guess
        if re.search(r"\b(in|on|at|to|for|with|of|by)\b", t):
            ptype = "PP"
        elif re.search(r"\b(is|are|was|were|am|be|been)\b", t):
            ptype = "VP_COP"
        elif re.search(r"\b(ran|run|made|make|see|saw|want|wanted)\b", t):
            ptype = "VP"
        else:
            ptype = "NP"
        out.append((t, ptype))
    return out


# ---------- main ----------
def build_or_load_index(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cache = load_cache(cfg["CACHE_JSON"])
    if cache:
        return cache
    pool = load_phrase_pool(cfg["PHRASE_POOL_CSV"])
    edges = load_edges(cfg["EDGES_CSV"])
    index = build_index(pool, edges)
    save_cache(index, cfg["CACHE_JSON"])
    return index

def score_sentence(sentence_or_phrases, ptypes: Optional[List[str]] = None, cfg: Dict[str, Any] = CONFIG):
    """
    If sentence_or_phrases is a string, we will chunk with tiny_chunk if USE_INTERNAL_CHUNKER is True.
    Else pass a list of (phrase_text, phrase_type).
    """
    index = build_or_load_index(cfg)

    if isinstance(sentence_or_phrases, str):
        if not cfg["USE_INTERNAL_CHUNKER"]:
            raise ValueError("Pass phrases because USE_INTERNAL_CHUNKER is False")
        phrases = tiny_chunk(sentence_or_phrases)
    else:
        phrases = sentence_or_phrases

    res = score_labels_for_phrases(phrases, index, cfg)
    # Pretty print short summary
    print("Phrase matches and contributions")
    for entry in res["per_phrase"]:
        if not entry["matches"]:
            print(f"  [{entry['type']}] {entry['phrase']}  -> no match")
            continue
        print(f"  [{entry['type']}] {entry['phrase']}")
        for m in entry["matches"][:3]:
            lab_str = ", ".join(f"{d['label']}:{d['score']:.2f}" for d in m["labels"][:3])
            print(f"    ~{m['sim']:.2f}  {m['key']}   {lab_str}")
    print("\nLabel scores")
    for L, s in sorted(res["label_scores"].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {L:20s}  {s:.3f}   P={res['label_probs'].get(L,0):.3f}")
    return res


if __name__ == "__main__":
    # Edit CONFIG here with your CSV locations, then try either:
    # 1) raw sentence with the tiny chunker (set USE_INTERNAL_CHUNKER=True)
    # 2) already chunked phrases from your splitter
    cfg = CONFIG

    # Example 1: use tiny chunker for a quick smoke test
    cfg["USE_INTERNAL_CHUNKER"] = True
    cfg["PHRASE_POOL_CSV"] = f"Phrase_out/out/-phrase_pool.csv"
    cfg["EDGES_CSV"] = f"Phrase_out/out/-phrase_label_edges.csv"
    cfg["CACHE_JSON"] = f"Phrase_out/out/cache_phrase_index.json"
    # Optional group boosts
    # cfg["KEYWORD_GROUPS"] = {"Scheduling": ["schedule", r"\bappointment\b"], "Apology": ["sorry", "apologize"]}

    test_sentence = "So sorry we ran out of time yesterday afternoon and the doctor is swamped"
    score_sentence(test_sentence, cfg=cfg)

    # Example 2: pass phrases from your chunker
    # phrases = [("we", "NP"), ("ran out of time", "VP_PHV"), ("yesterday afternoon", "ADVP")]
    # score_sentence(phrases, cfg=cfg)
