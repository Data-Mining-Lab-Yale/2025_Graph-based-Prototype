# stats_phrases_config.py
"""
Edit CONFIG once, then run:
    python stats_phrases_config.py
"""

import json
import math
import os
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple, Set

# Try pandas for CSVs. If not installed, the script will write JSON files instead.
try:
    import pandas as pd
except Exception:
    pd = None


# ========= CONFIG =========
CONFIG = {
    # One or many inputs. Each can be:
    # - JSON dict keyed by IDs
    # - merged final JSON dict
    # - checkpoint JSONL with lines {"id": ..., "record": {...}}
    "INPUTS": [
        f"Phrase_out/spacy_goal_toyset_raw.chunked.json",
        # r"/path/to/goal_toyset_raw.llm_all.partial.json",
        # r"/path/to/goal_toyset_raw.llm_all.checkpoint.jsonl",
    ],

    # Output folder for all stats files
    "OUTDIR": f"Phrase_out/out",

    # Which fields to read phrases from
    "FIELDS": ["text", "span"],

    # Filters for the phrase pool and edges
    "MIN_DOC_FREQ": 1,        # min number of records containing the phrase
    "MIN_OCCURRENCES": 1,     # min total occurrences across all records
    "MIN_PHRASE_LEN": 1,      # min character length for a phrase text
}
# ========= END CONFIG =========


def read_any(path: str) -> Dict[str, Any]:
    """
    Returns a dict keyed by id -> record.
    Accepts:
      - JSON dict keyed by IDs
      - merged final JSON dict
      - checkpoint JSONL {"id":..., "record": {...}}
    """
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "{":
            obj = json.load(f)
            if isinstance(obj, dict):
                return obj
            return {"__only__": obj}
        else:
            out: Dict[str, Any] = {}
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                rid = row.get("id")
                rec = row.get("record")
                if isinstance(rid, str) and isinstance(rec, dict):
                    out[rid] = rec
            return out


def load_many(paths: List[str]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for p in paths:
        part = read_any(p)
        merged.update(part)   # later files override earlier if same id
    return merged


def normalize_labels(rec: Dict[str, Any]) -> List[str]:
    """
    Uses selected_labels if present.
    If only labels exists as a list of objects with "label": [..], flatten it.
    """
    sel = rec.get("selected_labels")
    if isinstance(sel, list) and sel:
        return [str(x) for x in sel]
    labs = rec.get("labels")
    if isinstance(labs, list) and labs:
        bag: Set[str] = set()
        for obj in labs:
            if isinstance(obj, dict):
                for l in obj.get("label", []) or []:
                    bag.add(str(l))
        return sorted(bag)
    return []


def iter_phrases_from_record(rec: Dict[str, Any], fields: Tuple[str, ...]) -> Iterable[Tuple[str, str, str]]:
    """
    Yield (phrase_text, phrase_type, field) for every phrase in selected fields.
    Skips empty texts.
    """
    for field in fields:
        chunk = rec.get("chunks", {}).get(field, {})
        for sent in chunk.get("sentences", []) or []:
            for cl in sent.get("clauses", []) or []:
                for ph in cl.get("phrases", []) or []:
                    text = (ph.get("text") or "").strip()
                    ptype = (ph.get("type") or "OTHER").strip()
                    if text:
                        yield text, ptype, field


def safe_token_count(s: str) -> int:
    return len([t for t in s.split() if t])


def entropy_from_counts(label_counts: Dict[str, int]) -> float:
    total = sum(label_counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in label_counts.values():
        if c > 0:
            p = c / total
            ent -= p * math.log2(p)
    return round(ent, 6)


def analyze(records: Dict[str, Any],
            fields: Tuple[str, ...],
            min_doc_freq: int,
            min_occurrences: int,
            min_phrase_len: int) -> Dict[str, Any]:
    phrase_type_counter = Counter()
    phrase_type_by_label = defaultdict(Counter)
    phrase_type_by_label_field = defaultdict(lambda: defaultdict(Counter))
    phrase_text_type_counter = Counter()
    phrase_text_counter = Counter()
    label_counter = Counter()
    total_phrases = 0

    # document frequency maps
    phrase_docfreq: Counter = Counter()     # (text,type) -> #records containing it
    label_docfreq: Counter = Counter()      # label -> #records containing it
    pair_docfreq: Counter = Counter()       # ((text,type), label) -> #records containing both

    rec_phrase_sets: Dict[str, Set[Tuple[str, str]]] = {}
    rec_label_sets: Dict[str, Set[str]] = {}
    per_record_counts: Dict[str, Dict[str, int]] = {}

    ids = list(records.keys())
    for rid in ids:
        rec = records[rid]
        labs = set(normalize_labels(rec))
        rec_label_sets[rid] = labs
        for l in labs:
            label_counter[l] += 1
            label_docfreq[l] += 1

        sents = 0
        clauses = 0
        phrases = 0
        per_record_phrase_set: Set[Tuple[str, str]] = set()

        for field in fields:
            chunk = rec.get("chunks", {}).get(field, {})
            for sent in chunk.get("sentences", []) or []:
                sents += 1
                for cl in sent.get("clauses", []) or []:
                    clauses += 1
                    for ph in cl.get("phrases", []) or []:
                        text = (ph.get("text") or "").strip()
                        ptype = (ph.get("type") or "OTHER").strip()
                        if not text:
                            continue
                        if len(text) < min_phrase_len:
                            continue
                        phrases += 1
                        total_phrases += 1
                        phrase_type_counter[ptype] += 1
                        phrase_text_type_counter[(text, ptype)] += 1
                        phrase_text_counter[text] += 1
                        per_record_phrase_set.add((text, ptype))
                        for l in labs:
                            phrase_type_by_label[l][ptype] += 1
                            phrase_type_by_label_field[l][field][ptype] += 1

        rec_phrase_sets[rid] = per_record_phrase_set
        per_record_counts[rid] = {"sentences": sents, "clauses": clauses, "phrases": phrases}

    for rid, pset in rec_phrase_sets.items():
        labs = rec_label_sets.get(rid, set())
        for pt in pset:
            phrase_docfreq[pt] += 1
            for l in labs:
                pair_docfreq[(pt, l)] += 1

    N_records = len(ids)
    total_sents = sum(v["sentences"] for v in per_record_counts.values())
    total_clauses = sum(v["clauses"] for v in per_record_counts.values())
    total_phrases_sum = sum(v["phrases"] for v in per_record_counts.values())

    overview = [
        {"metric": "records", "value": N_records},
        {"metric": "labels_total", "value": sum(label_counter.values())},
        {"metric": "total_phrases", "value": total_phrases_sum},
        {"metric": "avg_phrases_per_record", "value": round(total_phrases_sum / max(N_records, 1), 4)},
        {"metric": "avg_phrases_per_sentence", "value": round(total_phrases_sum / max(total_sents, 1), 4)},
        {"metric": "avg_phrases_per_clause", "value": round(total_phrases_sum / max(total_clauses, 1), 4)},
    ]

    labels_sorted = sorted(label_counter.keys())
    ptypes_sorted = sorted(phrase_type_counter.keys())

    rows_types = [{"phrase_type": pt, "count": cnt} for pt, cnt in phrase_type_counter.items()]

    rows_type_by_label = []
    for lab in labels_sorted:
        for pt in ptypes_sorted:
            rows_type_by_label.append({"label": lab, "phrase_type": pt, "count": phrase_type_by_label[lab][pt]})

    rows_type_by_label_field = []
    for lab in labels_sorted:
        for field in fields:
            counts = phrase_type_by_label_field[lab][field]
            for pt in ptypes_sorted:
                rows_type_by_label_field.append({
                    "label": lab, "field": field, "phrase_type": pt, "count": counts[pt]
                })

    # phrase pool
    rows_pool = []
    for (text, ptype), occ in phrase_text_type_counter.items():
        dfreq = phrase_docfreq[(text, ptype)]
        if occ < min_occurrences or dfreq < min_doc_freq:
            continue
        lab_counts = {lab: pair_docfreq[((text, ptype), lab)] for lab in labels_sorted}
        p_phrase = dfreq / max(N_records, 1)
        best_label = None
        best_pmi = float("-inf")

        for lab in labels_sorted:
            c_xy = lab_counts[lab]
            if c_xy == 0 or dfreq == 0 or label_docfreq[lab] == 0:
                pmi = None
            else:
                p_xy = c_xy / N_records
                p_l = label_docfreq[lab] / N_records
                if p_phrase > 0 and p_l > 0 and p_xy > 0:
                    pmi = math.log2(p_xy / (p_phrase * p_l))
                else:
                    pmi = None
            lab_counts[lab] = {"docfreq": c_xy, "pmi": pmi}
            if pmi is not None and pmi > best_pmi:
                best_pmi = pmi
                best_label = lab

        lab_df_counts = {lab: lab_counts[lab]["docfreq"] for lab in labels_sorted}
        total_df = sum(lab_df_counts.values())
        purity = max(lab_df_counts.values()) / total_df if total_df > 0 else 0.0
        ent = entropy_from_counts(lab_df_counts)

        rows_pool.append({
            "phrase_text": text,
            "phrase_type": ptype,
            "occurrences": occ,
            "doc_freq": dfreq,
            "avg_len_char": round(len(text), 2),
            "avg_len_tokens": safe_token_count(text),
            "top_label_by_PMI": best_label,
            "top_label_PMI": round(best_pmi, 4) if math.isfinite(best_pmi) else None,
            "label_purity": round(purity, 4),
            "label_entropy": ent,
            **{f"label_{lab}_docfreq": lab_df_counts[lab] for lab in labels_sorted},
            **{f"label_{lab}_PMI": (round(lab_counts[lab]['pmi'], 4) if lab_counts[lab]['pmi'] is not None else None)
               for lab in labels_sorted},
        })

    # edges
    rows_edges = []
    for ((text, ptype), lab), c_xy in pair_docfreq.items():
        dfreq = phrase_docfreq[(text, ptype)]
        lfreq = label_docfreq[lab]
        if dfreq < min_doc_freq or phrase_text_type_counter[(text, ptype)] < min_occurrences:
            continue
        if N_records == 0 or dfreq == 0 or lfreq == 0:
            pmi = None
        else:
            p_xy = c_xy / N_records
            p_x = dfreq / N_records
            p_y = lfreq / N_records
            pmi = math.log2(p_xy / (p_x * p_y)) if p_x > 0 and p_y > 0 and p_xy > 0 else None
        rows_edges.append({
            "phrase_text": text,
            "phrase_type": ptype,
            "label": lab,
            "doc_cofreq": c_xy,
            "pmi": round(pmi, 4) if pmi is not None and math.isfinite(pmi) else None
        })

    rows_records = [{"id": rid, **per_record_counts[rid]} for rid in ids]

    return {
        "overview": overview,
        "phrase_types": rows_types,
        "phrase_type_by_label": rows_type_by_label,
        "phrase_type_by_label_field": rows_type_by_label_field,
        "phrase_pool": rows_pool,
        "edges": rows_edges,
        "record_counts": rows_records,
    }


def write_outputs(res: Dict[str, Any], outdir: str):
    os.makedirs(outdir, exist_ok=True)

    if pd is not None:
        pd.DataFrame(res["overview"]).to_csv(os.path.join(outdir, "stats_overview.csv"), index=False)
        pd.DataFrame(res["phrase_types"]).to_csv(os.path.join(outdir, "stats_phrase_types.csv"), index=False)
        pd.DataFrame(res["phrase_type_by_label"]).to_csv(os.path.join(outdir, "stats_phrase_type_by_label.csv"), index=False)
        pd.DataFrame(res["phrase_type_by_label_field"]).to_csv(os.path.join(outdir, "stats_phrase_type_by_label_field.csv"), index=False)
        pd.DataFrame(res["phrase_pool"]).to_csv(os.path.join(outdir, "phrase_pool.csv"), index=False)
        pd.DataFrame(res["edges"]).to_csv(os.path.join(outdir, "phrase_label_edges.csv"), index=False)
        pd.DataFrame(res["record_counts"]).to_csv(os.path.join(outdir, "record_level_counts.csv"), index=False)
    else:
        # fallback to JSON if pandas is unavailable
        for name in ["overview", "phrase_types", "phrase_type_by_label", "phrase_type_by_label_field",
                     "phrase_pool", "edges", "record_counts"]:
            path = os.path.join(outdir, name + ".json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(res[name], f, ensure_ascii=False, indent=2)

    # quick JSON sample for dashboards
    quick = {
        "overview": res["overview"],
        "top_types": sorted(res["phrase_types"], key=lambda r: r["count"], reverse=True)[:20],
        "top_phrases_overall": sorted(res["phrase_pool"],
                                      key=lambda r: (r["doc_freq"], r["occurrences"]),
                                      reverse=True)[:30],
    }
    with open(os.path.join(outdir, "stats_quick.json"), "w", encoding="utf-8") as f:
        json.dump(quick, f, ensure_ascii=False, indent=2)


def main():
    cfg = CONFIG
    inputs = cfg["INPUTS"]
    outdir = cfg["OUTDIR"]
    fields = tuple(cfg["FIELDS"])
    min_doc = int(cfg["MIN_DOC_FREQ"])
    min_occ = int(cfg["MIN_OCCURRENCES"])
    min_len = int(cfg["MIN_PHRASE_LEN"])

    merged = load_many(inputs)
    res = analyze(merged, fields, min_doc, min_occ, min_len)
    write_outputs(res, outdir)
    print(f"Done. Wrote stats to {outdir}")


if __name__ == "__main__":
    main()
