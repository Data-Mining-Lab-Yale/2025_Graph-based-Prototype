import json
import os
import re
from typing import List, Dict, Any

# -------- configuration --------
INPUT_FILES = [
    "EPPC_output_json/CleanedData/processed_annotations_with_types.json",
    "EPPC_output_json/sentence_goal_oriented_label.json",
    "EPPC_output_json/sentence_interactional_label.json",
    "EPPC_output_json/subsentence_goal_oriented_label.json",
    "EPPC_output_json/subsentence_interactional_label.json",
]
OUTPUT_DIR = "split_by_clause"
MIN_WORDS_PER_SEGMENT = 3
MIN_CLAUSE_SEGMENTS_FOR_CROSS = 3
# --------------------------------

# Clause boundary pattern:
# - sentence enders: . ! ? ; :
# - commas and dashes used as separators
# - common conjunctions
BOUNDARY_RE = re.compile(
    r"""
    [.!?;:]+                             # sentence enders
    | \s*[,-]\s*                         # comma or dash with spaces
    | \s+\b(?:and|but|or|so|because|while|although|though|
             however|therefore|moreover|whereas|meanwhile|plus)\b\s+
    """,
    re.IGNORECASE | re.VERBOSE,
)

def count_clause_like_segments(text: str) -> int:
    """Split text on boundary markers and count segments with at least MIN_WORDS_PER_SEGMENT words."""
    # Normalize spaces
    t = re.sub(r"\s+", " ", (text or "")).strip()
    if not t:
        return 0
    # Split
    segments = [s.strip() for s in BOUNDARY_RE.split(t) if s and s.strip()]
    # Count segments that look clause like
    strong_segments = [s for s in segments if len(s.split()) >= MIN_WORDS_PER_SEGMENT]
    return len(strong_segments)

def is_cross_clause(span: str) -> bool:
    return count_clause_like_segments(span) >= MIN_CLAUSE_SEGMENTS_FOR_CROSS

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def split_processed_annotations(data: List[Dict[str, Any]]):
    """Input is a list of message dicts with an 'annotations' list.
       The span to test is annotations[i]['text'].
       Output two lists with the same schema, but each message keeps only the annotations that match."""
    only_cross = []
    no_cross = []

    for msg in data:
        anns = msg.get("annotations", [])
        cross_anns = []
        pure_anns = []
        for ann in anns:
            span_text = ann.get("text", "")
            if is_cross_clause(span_text):
                cross_anns.append(ann)
            else:
                pure_anns.append(ann)

        if cross_anns:
            m = dict(msg)
            m["annotations"] = cross_anns
            only_cross.append(m)
        if pure_anns:
            m = dict(msg)
            m["annotations"] = pure_anns
            no_cross.append(m)

    return only_cross, no_cross

def split_label_dict(data: Dict[str, Any]):
    """Input is a dict of id -> {text, span, labels}.
       The span to test is item['span'].
       Output two dicts with the same mapping keys as selected."""
    only_cross = {}
    no_cross = {}

    for k, item in data.items():
        span_text = item.get("span", "")
        if is_cross_clause(span_text):
            only_cross[k] = item
        else:
            no_cross[k] = item

    return only_cross, no_cross

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary = []

    for fname in INPUT_FILES:
        if not os.path.exists(fname):
            summary.append(f"{fname}: not found, skipped")
            continue

        data = load_json(fname)

        if fname == "processed_annotations_with_types.json" or "EPPC_output_json/CleanedData/processed_annotations_with_types.json":
            cross, pure = split_processed_annotations(data)
        else:
            cross, pure = split_label_dict(data)

        base = os.path.splitext(os.path.basename(fname))[0]
        out_cross = os.path.join(OUTPUT_DIR, f"{base}__only_cross_clause.json")
        out_pure = os.path.join(OUTPUT_DIR, f"{base}__no_cross_clause.json")

        save_json(cross, out_cross)
        save_json(pure, out_pure)

        # Simple counts
        if fname == "processed_annotations_with_types.json":
            n_items = sum(len(m.get("annotations", [])) for m in data)
            n_cross = sum(len(m.get("annotations", [])) for m in cross)
            n_pure = sum(len(m.get("annotations", [])) for m in pure)
            summary.append(
                f"{fname}: total annotations {n_items} | cross {n_cross} | no cross {n_pure}"
            )
        else:
            n_items = len(data)
            n_cross = len(cross)
            n_pure = len(pure)
            summary.append(
                f"{fname}: total entries {n_items} | cross {n_cross} | no cross {n_pure}"
            )

    print("\nSplit complete. Files saved in:", OUTPUT_DIR)
    print("\nSummary:")
    for line in summary:
        print(" -", line)

if __name__ == "__main__":
    main()
