"""
Split annotations by cross clause spans and flag long spans, preserving original schemas.

Outputs per input:
  <base>__only_cross_clause.json
  <base>__no_cross_clause.json
  <base>__span_flags.json
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

# --------- thresholds you can adjust ----------
MIN_WORDS_PER_SEGMENT = 3
MIN_CLAUSE_SEGMENTS_FOR_CROSS = 3
LONG_TOKEN_THRESHOLD = 11
# ----------------------------------------------

# Clause boundary pattern
BOUNDARY_RE = re.compile(
    r"""
    [.!?;:]+
    | \s*[,-]\s*
    | \s+\b(?:and|but|or|so|because|while|although|though|
             however|therefore|moreover|whereas|meanwhile|plus)\b\s+
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Tokenizer similar to a conservative word plus punct split
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall((text or "").strip())


def count_clause_like_segments(text: str) -> int:
    t = re.sub(r"\s+", " ", (text or "")).strip()
    if not t:
        return 0
    segs = [s.strip() for s in BOUNDARY_RE.split(t) if s and s.strip()]
    strong = [s for s in segs if len(s.split()) >= MIN_WORDS_PER_SEGMENT]
    return len(strong)


def compute_flags(span_text: str) -> Dict[str, Any]:
    toks = tokenize(span_text)
    n_tokens = len(toks)
    n_clauses = count_clause_like_segments(span_text)
    is_long = n_tokens >= LONG_TOKEN_THRESHOLD
    is_cross = n_clauses >= MIN_CLAUSE_SEGMENTS_FOR_CROSS
    return {
        "span_len_tokens": n_tokens,
        "num_clause_segments": n_clauses,
        "is_long": is_long,
        "is_cross": is_cross,
    }


def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, p: Path):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def process_processed_annotations(inp: Path, outdir: Path) -> Tuple[int, int, int]:
    """
    Schema:
    [
      {
        "message": "...",
        "message_id": 1,
        "annotations": [
          {"text_id": "1_0", "text": "<SPAN>", "code": [...], "label_type": [...]},
          ...
        ]
      }, ...
    ]
    """
    data: List[Dict[str, Any]] = load_json(inp)
    cross_msgs: List[Dict[str, Any]] = []
    pure_msgs: List[Dict[str, Any]] = []
    flags: List[Dict[str, Any]] = []

    total_anns = 0
    cross_anns = 0
    pure_anns = 0

    for msg in data:
        anns = msg.get("annotations", []) or []
        if not anns:
            continue

        cross_list = []
        pure_list = []

        for ann in anns:
            span_text = ann.get("text", "")
            f = compute_flags(span_text)
            flags.append({
                "message_id": msg.get("message_id"),
                "text_id": ann.get("text_id"),
                "span_len_tokens": f["span_len_tokens"],
                "num_clause_segments": f["num_clause_segments"],
                "is_long": f["is_long"],
                "is_cross": f["is_cross"],
            })

            total_anns += 1
            if f["is_cross"]:
                cross_list.append(ann)
                cross_anns += 1
            else:
                pure_list.append(ann)
                pure_anns += 1

        if cross_list:
            m = dict(msg)
            m["annotations"] = cross_list
            cross_msgs.append(m)
        if pure_list:
            m = dict(msg)
            m["annotations"] = pure_list
            pure_msgs.append(m)

    base = inp.stem
    save_json(cross_msgs, outdir / f"{base}__only_cross_clause.json")
    save_json(pure_msgs, outdir / f"{base}__no_cross_clause.json")
    save_json(flags, outdir / f"{base}__span_flags.json")

    return total_anns, cross_anns, pure_anns


def process_label_dict(inp: Path, outdir: Path) -> Tuple[int, int, int]:
    """
    Schema:
    {
      "<id>": {"text": "...", "span": "<SPAN>", "labels": [...]},
      ...
    }
    """
    data: Dict[str, Any] = load_json(inp)
    only_cross: Dict[str, Any] = {}
    no_cross: Dict[str, Any] = {}
    flags: List[Dict[str, Any]] = []

    total = 0
    cross_n = 0
    pure_n = 0

    for k, item in data.items():
        span_text = item.get("span", "")
        f = compute_flags(span_text)
        flags.append({
            "id": k,
            "span_len_tokens": f["span_len_tokens"],
            "num_clause_segments": f["num_clause_segments"],
            "is_long": f["is_long"],
            "is_cross": f["is_cross"],
        })

        total += 1
        if f["is_cross"]:
            only_cross[k] = item
            cross_n += 1
        else:
            no_cross[k] = item
            pure_n += 1

    base = inp.stem
    save_json(only_cross, outdir / f"{base}__only_cross_clause.json")
    save_json(no_cross, outdir / f"{base}__no_cross_clause.json")
    save_json(flags, outdir / f"{base}__span_flags.json")

    return total, cross_n, pure_n


def main():
    # declare globals before first use in this scope
    global MIN_WORDS_PER_SEGMENT, MIN_CLAUSE_SEGMENTS_FOR_CROSS, LONG_TOKEN_THRESHOLD

    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", action="append", default=None,
                    help="Input JSON file. Repeat this flag for multiple files.")
    ap.add_argument("--outdir", type=str, default=None,
                    help="Output folder. Will be created if missing.")
    ap.add_argument("--min_words_per_segment", type=int, default=MIN_WORDS_PER_SEGMENT)
    ap.add_argument("--min_segments_for_cross", type=int, default=MIN_CLAUSE_SEGMENTS_FOR_CROSS)
    ap.add_argument("--long_token_threshold", type=int, default=LONG_TOKEN_THRESHOLD)

    args = ap.parse_args()

    # defaults if not provided
    if not args.inputs:
        args.inputs = [
            "EPPC_output_json/CleanedData/processed_annotations_with_types.json",
            "EPPC_output_json/sentence_goal_oriented_label.json",
            "EPPC_output_json/sentence_interactional_label.json",
            "EPPC_output_json/subsentence_goal_oriented_label.json",
            "EPPC_output_json/subsentence_interactional_label.json",
        ]
    if not args.outdir:
        args.outdir = "EPPC_output_json/cross_clause_splits"

    # update thresholds from CLI
    MIN_WORDS_PER_SEGMENT = args.min_words_per_segment
    MIN_CLAUSE_SEGMENTS_FOR_CROSS = args.min_segments_for_cross
    LONG_TOKEN_THRESHOLD = args.long_token_threshold

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary = []
    for in_path in args.inputs:
        p = Path(in_path)
        if not p.exists():
            summary.append(f"{p} : missing")
            continue

        # pick handler based on schema by filename
        if p.name == "processed_annotations_with_types.json":
            total, cross_n, pure_n = process_processed_annotations(p, outdir)
            summary.append(f"{p.name}: annotations total={total}, cross={cross_n}, no_cross={pure_n}")
        else:
            total, cross_n, pure_n = process_label_dict(p, outdir)
            summary.append(f"{p.name}: entries total={total}, cross={cross_n}, no_cross={pure_n}")

    print("\nDone. Outputs in:", outdir)
    print("Summary:")
    for line in summary:
        print(" -", line)


if __name__ == "__main__":
    main()
