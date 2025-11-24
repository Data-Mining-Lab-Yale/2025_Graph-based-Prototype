# Dataset_Stat_4_SpanConsistency.py
# Span inconsistency analysis with zero made-up numbers.
# Inputs:
#   - processed_messages_with_annotations.json
#   - annotation_code_mapping_detailed_corrected.json
#
# Outputs (in span_stats/):
#   - span_stats.csv                        (per-annotation rows)
#   - overall_span_length_stats.json        (summary)
#   - per_label_span_stats.csv              (per-label summary)
#   - span_length_histogram.png             (overall token-length histogram)
#   - span_length_buckets.csv               (bucket counts overall + per label)
#   - alignment_summary.json                (alignment proportions or placeholder note)
#   - boundary_distance_stats.json          (token distances to nearest sentence boundary)
#   - latex_span_inconsistency_snippet.tex  (LaTeX-ready with computed numbers or [NOT_AVAILABLE])

import os, json, math, csv, statistics, re
from collections import defaultdict, Counter
from typing import Optional, Tuple, List

import matplotlib
matplotlib.use("Agg")  # safe in headless
import matplotlib.pyplot as plt

# ========= Config (edit paths if needed) =========
ANNOTATION_FILE = "EPPC_output_json/CleanedData/processed_messages_with_annotations.json"
MAPPING_FILE    = "EPPC_output_json/Labels/annotation_code_mapping_detailed_corrected.json"
OUTDIR          = "Data_for_Evidences/span_stats"

# Text fields to try (first found wins)
MESSAGE_TEXT_FIELDS = [
    "message",          # <-- your field
    "message_text", "text", "content", "raw_text", "msg_text"
]
SPAN_TEXT_FIELDS = [
    "text",             # <-- your field (annotation-level)
    "span_text", "clause_text", "segment_text", "content"
]

# Potential offset fields (if present)
OFFSET_FIELDS = [
    ("span_start", "span_end"),
    ("start", "end"),
    ("char_start", "char_end"),
    ("offset_start", "offset_end"),
]

# Token length buckets (inclusive ranges; last is open-ended)
BUCKETS = [(1,2), (3,5), (6,10), (11,20), (21,999999)]

# Histogram binning
HIST_BINS = list(range(1, 51)) + [60, 80, 120, 200]
DPI = 200

# ========= IO prep =========
os.makedirs(OUTDIR, exist_ok=True)
with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    MESSAGES = json.load(f)

if os.path.exists(MAPPING_FILE):
    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        MAPPING = json.load(f)
else:
    MAPPING = {}

# ========= Helpers =========
def simple_tokenize(s: str) -> List[str]:
    # conservative tokenization: words, long dashes, and other single non-space chars
    return re.findall(r"[A-Za-z0-9']+|-{2,}|[^\sA-Za-z0-9']", s)

def choose_message_text(msg: dict) -> Optional[str]:
    for k in MESSAGE_TEXT_FIELDS:
        v = msg.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None

def choose_span_text(ann: dict) -> Optional[str]:
    for k in SPAN_TEXT_FIELDS:
        v = ann.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

def get_offsets(ann: dict) -> Optional[Tuple[int,int]]:
    for a,b in OFFSET_FIELDS:
        if a in ann and b in ann:
            try:
                start = int(ann[a]); end = int(ann[b])
                if end < start:
                    return None
                # normalize inclusive end to exclusive
                if "inclusive" in b.lower() or (end - start) < 0:
                    end = end + 1
                return start, end
            except Exception:
                continue
    return None

def map_tokens_to_label(ann: dict) -> Optional[str]:
    """Prefer subcode, else code — using annotation_code_mapping_detailed_corrected.json if present."""
    raw = ann.get("code", [])
    if not isinstance(raw, list):
        return None
    code_label, subcode_label = None, None
    for tok in raw:
        info = MAPPING.get(tok) or MAPPING.get(str(tok).lower()) or MAPPING.get(str(tok).upper())
        if not info:
            continue
        level = info.get("level")
        matched = info.get("matched_codebook_label")
        if level == "code" and code_label is None:
            code_label = matched
        elif level == "subcode" and subcode_label is None:
            subcode_label = matched
    return subcode_label if subcode_label else code_label

# -------- robust fallback matching (no offsets) --------
def normalize_for_match(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9'\s]", " ", s)   # drop punctuation to spaces (keeps apostrophes)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def safe_find_span_offsets(span_text: str, message_text: str) -> Optional[Tuple[int,int]]:
    """Try exact match; if not found, use normalized match and map back to original."""
    if not span_text or not message_text:
        return None

    # 1) exact substring
    idx = message_text.find(span_text)
    if idx != -1:
        return idx, idx + len(span_text)

    # 2) normalized match
    norm_msg  = normalize_for_match(message_text)
    norm_span = normalize_for_match(span_text)
    j = norm_msg.find(norm_span)
    if j == -1:
        return None

    # 3) map back: scan corridor in original text near j
    approx_len = max(len(span_text) - 4, len(norm_span))
    best = None
    for start in range(max(0, j - 40), min(len(message_text), j + 40)):
        end = min(len(message_text), start + approx_len + 80)
        window = message_text[start:end]
        if normalize_for_match(window).startswith(norm_span):
            # tighten to shortest window that still matches
            trim_end = end
            while trim_end > start and normalize_for_match(message_text[start:trim_end]).startswith(norm_span):
                trim_end -= 1
            best = (start, trim_end + 1)
            break
    return best

# -------- sentence splitting (supports enumerations like '1) ... 2) ...') --------
def sentence_split_simple(text: str) -> List[Tuple[int,int]]:
    breaks = [0]
    for m in re.finditer(r"\b\d+\)\s+", text):           # enumeration bullets
        breaks.append(m.start())
    for m in re.finditer(r"[\.!\?]+(\s+|$)", text):      # standard end punctuation
        breaks.append(m.end())
    breaks = sorted(set([b for b in breaks if 0 <= b <= len(text)] + [len(text)]))

    spans = []
    for i in range(len(breaks) - 1):
        s, e = breaks[i], breaks[i+1]
        while s < e and text[s].isspace(): s += 1
        while e > s and text[e-1].isspace(): e -= 1
        if e > s:
            spans.append((s, e))
    return spans if spans else [(0, len(text))]

def in_any_span(target: Tuple[int,int], spans: List[Tuple[int,int]]) -> str:
    s, e = target
    for ss, ee in spans:
        if s == ss and e == ee: return "aligned"
        if s >= ss and e <= ee: return "inside"
        if (s < ss < e) or (s < ee < e) or (ss < s < ee) or (ss < e < ee):
            return "crosses"
    return "outside"

def iqr(values: List[float]) -> Optional[float]:
    if len(values) < 4: return None
    q1 = statistics.quantiles(values, n=4)[0]
    q3 = statistics.quantiles(values, n=4)[2]
    return q3 - q1

def coeff_var(values: List[float]) -> Optional[float]:
    if not values: return None
    mean = statistics.mean(values)
    if mean == 0: return None
    return statistics.pstdev(values) / mean

def bucket_name(n_tok: int) -> str:
    for lo, hi in BUCKETS:
        if lo <= n_tok <= hi:
            return f"{lo}-{hi if hi < 999999 else '+'}"
    return "other"

def token_boundaries(text: str, span: Tuple[int,int]) -> Tuple[int,int]:
    """Token distances from span start/end to nearest sentence boundary within the containing sentence."""
    sents = sentence_split_simple(text)
    s, e = span
    container = None
    for ss, ee in sents:
        if s >= ss and e <= ee:
            container = (ss, ee)
            break
    if container is None:
        # choose sentence with max overlap
        best, best_ov = None, -1
        for ss, ee in sents:
            ov = max(0, min(e, ee) - max(s, ss))
            if ov > best_ov:
                best_ov, best = ov, (ss, ee)
        container = best if best else (0, len(text))
    ss, ee = container
    left_text  = text[ss:s]
    right_text = text[e:ee]
    return len(simple_tokenize(left_text)), len(simple_tokenize(right_text))

# ========= Pass 1: collect per-annotation stats =========
rows = []
length_tokens, length_chars = [], []
per_label_tokens = defaultdict(list)
bucket_counts = Counter()
bucket_counts_by_label = defaultdict(Counter)

alignment_counts = Counter()
alignment_counts_by_label = defaultdict(Counter)
alignment_possible = False

for msg_idx, msg in enumerate(MESSAGES):
    msg_text = choose_message_text(msg)  # may be None
    for ann_idx, ann in enumerate(msg.get("annotations", [])):
        span = choose_span_text(ann)
        label = ann.get("label") or map_tokens_to_label(ann) or "[UNMAPPED]"
        if not span or not isinstance(span, str):
            continue

        # lengths
        n_tok = len(simple_tokenize(span))
        n_chr = len(span)
        length_tokens.append(n_tok)
        length_chars.append(n_chr)
        per_label_tokens[label].append(n_tok)

        bname = bucket_name(n_tok)
        bucket_counts[bname] += 1
        bucket_counts_by_label[label][bname] += 1

        # Alignment + boundary distances
        relation = "[NOT_AVAILABLE]"
        used_offsets = False
        boundary_left = None
        boundary_right = None

        if msg_text:
            pos = get_offsets(ann)
            if pos:
                used_offsets = True
            else:
                pos = safe_find_span_offsets(span, msg_text)

            if pos:
                alignment_possible = True
                sent_spans = sentence_split_simple(msg_text)
                relation = in_any_span(pos, sent_spans)
                bl, br = token_boundaries(msg_text, pos)
                boundary_left, boundary_right = bl, br

        rows.append({
            "message_index": msg_idx,
            "annotation_index": ann_idx,
            "label": label,
            "span_text": span,
            "length_tokens": n_tok,
            "length_chars": n_chr,
            "bucket": bname,
            "alignment_relation": relation,
            "alignment_offsets_used": used_offsets,
            "boundary_left_tokens": boundary_left,
            "boundary_right_tokens": boundary_right,
        })

        alignment_counts[relation] += 1
        alignment_counts_by_label[label][relation] += 1

# ========= Save per-annotation CSV =========
per_ann_csv = os.path.join(OUTDIR, "span_stats.csv")
with open(per_ann_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
        "message_index","annotation_index","label","span_text","length_tokens","length_chars",
        "bucket","alignment_relation","alignment_offsets_used","boundary_left_tokens","boundary_right_tokens"
    ])
    w.writeheader()
    for r in rows:
        w.writerow(r)

# ========= Overall length stats =========
overall = {}
if length_tokens:
    overall.update({
        "token_min": min(length_tokens),
        "token_max": max(length_tokens),
        "token_median": float(statistics.median(length_tokens)),
        "token_mean": float(statistics.mean(length_tokens)),
        "token_iqr": float(iqr(length_tokens)) if iqr(length_tokens) is not None else None,
        "token_cv":  float(coeff_var(length_tokens)) if coeff_var(length_tokens) is not None else None,
    })
else:
    overall.update({k: None for k in ["token_min","token_max","token_median","token_mean","token_iqr","token_cv"]})

if length_chars:
    overall.update({
        "char_min": min(length_chars),
        "char_max": max(length_chars),
        "char_median": float(statistics.median(length_chars)),
        "char_mean": float(statistics.mean(length_chars)),
        "char_iqr": float(iqr(length_chars)) if iqr(length_chars) is not None else None,
        "char_cv":  float(coeff_var(length_chars)) if coeff_var(length_chars) is not None else None,
    })
else:
    overall.update({k: None for k in ["char_min","char_max","char_median","char_mean","char_iqr","char_cv"]})

with open(os.path.join(OUTDIR, "overall_span_length_stats.json"), "w", encoding="utf-8") as f:
    json.dump(overall, f, indent=2, ensure_ascii=False)

# ========= Per-label length stats =========
per_label_rows = []
for lab, toks_list in per_label_tokens.items():
    d = {"label": lab, "n": len(toks_list)}
    if toks_list:
        d.update({
            "token_min": int(min(toks_list)),
            "token_median": float(statistics.median(toks_list)),
            "token_mean": float(statistics.mean(toks_list)),
            "token_max": int(max(toks_list)),
            "token_iqr": float(iqr(toks_list)) if iqr(toks_list) is not None else None,
            "token_cv":  float(coeff_var(toks_list)) if coeff_var(toks_list) is not None else None,
        })
    else:
        d.update({"token_min": None, "token_median": None, "token_mean": None, "token_max": None, "token_iqr": None, "token_cv": None})
    total = sum(bucket_counts_by_label[lab].values()) or 1
    for lo, hi in BUCKETS:
        b = f"{lo}-{hi if hi<999999 else '+'}"
        d[f"bucket_{b}"] = float(bucket_counts_by_label[lab][b] / total)
    per_label_rows.append(d)

per_label_csv = os.path.join(OUTDIR, "per_label_span_stats.csv")
all_cols = ["label","n","token_min","token_median","token_mean","token_max","token_iqr","token_cv"] + \
           [f"bucket_{lo}-{hi if hi<999999 else '+'}" for lo,hi in BUCKETS]
with open(per_label_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=all_cols)
    w.writeheader()
    for d in per_label_rows:
        w.writerow(d)

# ========= Bucket table overall =========
with open(os.path.join(OUTDIR, "span_length_buckets.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["bucket","count"])
    for lo, hi in BUCKETS:
        bname = f"{lo}-{hi if hi<999999 else '+'}"
        w.writerow([bname, bucket_counts[bname]])

# ========= Alignment summary =========
align_summary = {}
if any(k != "[NOT_AVAILABLE]" for k in alignment_counts):
    total = sum(alignment_counts.values())
    def prop(k): 
        return float(alignment_counts[k]/total) if total>0 else 0.0
    align_summary = {
        "available": True,
        "counts": dict(alignment_counts),
        "proportions": {
            "aligned": prop("aligned"),
            "inside":  prop("inside"),
            "crosses": prop("crosses"),
            "outside": prop("outside"),
            "[NOT_AVAILABLE]": prop("[NOT_AVAILABLE]"),
        }
    }
else:
    align_summary = {
        "available": False,
        "note": "Message text and/or offsets not accessible under expected keys; alignment not computed."
    }

with open(os.path.join(OUTDIR, "alignment_summary.json"), "w", encoding="utf-8") as f:
    json.dump(align_summary, f, indent=2, ensure_ascii=False)

# ========= Boundary distance stats =========
bd_left  = [r["boundary_left_tokens"]  for r in rows if r["boundary_left_tokens"]  is not None]
bd_right = [r["boundary_right_tokens"] for r in rows if r["boundary_right_tokens"] is not None]
bd_all   = bd_left + bd_right
boundary_stats = {
    "available": bool(bd_all),
    "left_mean":   float(statistics.mean(bd_left))   if bd_left  else None,
    "right_mean":  float(statistics.mean(bd_right))  if bd_right else None,
    "left_median": float(statistics.median(bd_left)) if bd_left  else None,
    "right_median":float(statistics.median(bd_right))if bd_right else None,
    "overall_median": float(statistics.median(bd_all)) if bd_all else None
}
with open(os.path.join(OUTDIR, "boundary_distance_stats.json"), "w", encoding="utf-8") as f:
    json.dump(boundary_stats, f, indent=2, ensure_ascii=False)

# ========= Plot: span length histogram (tokens) =========
if length_tokens:
    plt.figure(figsize=(8,5))
    plt.hist(length_tokens, bins=HIST_BINS)
    plt.xlabel("Span length (tokens)")
    plt.ylabel("Count")
    plt.title("Distribution of annotated span lengths (tokens)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "span_length_histogram.png"), dpi=DPI)
    plt.close()

# ========= LaTeX snippet (placeholders where unavailable) =========
def fmt(v):
    if v is None: return "[NOT_AVAILABLE]"
    if isinstance(v, float): return f"{v:.3f}"
    return str(v)

token_min    = fmt(overall.get("token_min"))
token_max    = fmt(overall.get("token_max"))
token_median = fmt(overall.get("token_median"))
token_mean   = fmt(overall.get("token_mean"))
token_iqr    = fmt(overall.get("token_iqr"))
token_cv     = fmt(overall.get("token_cv"))

if align_summary.get("available"):
    props = align_summary["proportions"]
    p_aligned = f"{props.get('aligned',0)*100:.1f}\\%"
    p_inside  = f"{props.get('inside',0)*100:.1f}\\%"
    p_crosses = f"{props.get('crosses',0)*100:.1f}\\%"
    p_outside = f"{props.get('outside',0)*100:.1f}\\%"
else:
    p_aligned = p_inside = p_crosses = p_outside = "[NOT_AVAILABLE]"

# latex = f"""% Auto-generated by Dataset_Stat_4_SpanConsistency.py
# \\subsection{{Span Inconsistency}}

# Another challenge arises from \\textbf{{span inconsistency}} in the annotations.
# While intent labels are applied to text spans, these spans vary widely in length and their alignment with syntactic clauses is often irregular.
# Some annotations cover a single short phrase (e.g., ``thanks''), while others span multiple sentences or fragments that cross syntactic boundaries.
# This variability makes it difficult to establish a reliable clause-level unit of analysis and introduces noise into model training.

# \\paragraph{{Quantitative findings.}}
# We measured span statistics across the dataset:
# \\begin{{itemize}}
#     \\item \\textbf{{Length variation.}} Spans range from \\textit{{[{token_min}]}} to \\textit{{[{token_max}]}} tokens, with a median of \\textit{{[{token_median}]}} and mean of \\textit{{[{token_mean}]}} (IQR \\textit{{[{token_iqr}]}}, CV \\textit{{[{token_cv}]}}).
#     \\item \\textbf{{Sentence alignment.}} Approximately \\textit{{[{p_aligned}]}} of spans align exactly to sentence boundaries, while \\textit{{[{p_inside}]}} fall inside a sentence, \\textit{{[{p_crosses}]}} cross boundaries, and \\textit{{[{p_outside}]}} do not align cleanly.\\footnote{{Computed only when message text and offsets or a reliable substring match are available; else placeholders are shown.}}
#     \\item \\textbf{{Label distribution.}} Per-label span lengths and bucket proportions are in \\texttt{{{os.path.basename(per_label_csv)}}}.
# \\end{{itemize}}

# \\begin{{figure}}[h]
#     \\centering
#     \\includegraphics[width=0.7\\textwidth]{{{os.path.join(OUTDIR, 'span_length_histogram.png').replace('\\', '/')}}}
#     \\caption{{Distribution of annotated span token lengths.}}
#     \\label{{fig:span_length}}
# \\end{{figure}}
# """

# with open(os.path.join(OUTDIR, "latex_span_inconsistency_snippet.tex"), "w", encoding="utf-8") as f:
#     f.write(latex)

print("[DONE] Wrote outputs to:", OUTDIR)
