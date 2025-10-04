# Decision_Structure_1_chucking_checkpoint.py
# Resume-friendly chunking driver with checkpointing and validation
# Run: python Decision_Structure_1_chucking_checkpoint.py
# Edit CONFIG below for paths and backend. No flags needed.

import os, json, importlib
from typing import Any, Dict, List, Tuple, Optional

# ========= USER CONFIG =========
CONFIG = {
    # paths
    "input_json": "toysets_out/goal_toyset_raw.json",
    "output_dir": "Phrase_out",                 # None = next to input
    "final_out":  None,                         # None -> <base>.chunked.json
    "checkpoint": None,                         # None -> <base>.checkpoint.jsonl

    # what to process
    "fields": ["text", "span"],
    "backend": "llm",                           # "spacy" or "llm"
    "probe": 0,                                 # 0 to skip probe
    "stop_on_error": True,                      # stop if probe has any errors

    # spaCy backend
    "spacy_module": "Decision_Structure_1_chucking_v1",
    "spacy_fn": "split_text_into_clauses_and_phrases",

    # LLM backend (LM Studio module you have)
    "llm_module": "Decision_Structure_1_chucking_local_llm",
    "llm_fn": "lmstudio_chunk",
    # Optional overrides if your LLM module exposes these globals:
    "llm_base_url": "http://127.0.0.1:1234/v1",
    "llm_model": "openai/gpt-oss-20b",
    "llm_api_key": "lm-studio",
}
# ========= END CONFIG =========


# -------- validation --------
def non_overlapping(spans: List[Tuple[int, int]]) -> bool:
    spans = sorted(spans)
    for i in range(1, len(spans)):
        if spans[i][0] < spans[i - 1][1]:
            return False
    return True


def _clamp(a: int, lo: int, hi: int) -> int:
    return max(lo, min(a, hi))


def repair_field_result(fr: Any) -> Dict[str, Any]:
    """
    Make model output safe:
      - ensure 0 <= start < end <= len(input)
      - if text mismatch, try to realign spans by searching the text
      - if still bad, drop the bad span
    Always returns {"input": str, "sentences": [...]}
    """
    if not isinstance(fr, dict):
        return {"input": "", "sentences": []}
    if "input" not in fr or "sentences" not in fr:
        return {"input": fr.get("input", ""), "sentences": []}

    src = fr.get("input") or ""
    n = len(src)
    fixed_sents = []
    for s in fr.get("sentences", []):
        st = int(s.get("start", 0)); en = int(s.get("end", 0)); txt = s.get("text", "")
        st = _clamp(st, 0, n); en = _clamp(en, 0, n)
        if st >= en or not txt or src[st:en] != txt:
            if txt:
                pos = src.find(txt)
                if pos != -1:
                    st, en = pos, pos + len(txt)
                else:
                    continue
            else:
                continue

        fixed_clauses = []
        for c in s.get("clauses", []):
            cst = int(c.get("start", st)); cen = int(c.get("end", en)); ctxt = c.get("text", "")
            cst = _clamp(cst, st, en); cen = _clamp(cen, st, en)
            if cst >= cen or not ctxt or src[cst:cen] != ctxt:
                if ctxt:
                    pos = src.find(ctxt, st, en)
                    if pos != -1:
                        cst, cen = pos, pos + len(ctxt)
                    else:
                        continue
                else:
                    continue

            fixed_phr = []
            last_end = cst
            for p in c.get("phrases", []):
                pst = int(p.get("start", cst)); pen = int(p.get("end", cst)); ptxt = p.get("text", "")
                pst = _clamp(pst, cst, cen); pen = _clamp(pen, cst, cen)
                if not ptxt:
                    continue
                if src[pst:pen] != ptxt:
                    pos = src.find(ptxt, cst, cen)
                    if pos != -1:
                        pst, pen = pos, pos + len(ptxt)
                    else:
                        continue
                if pst < last_end:
                    continue
                fixed_phr.append({
                    "type": p.get("type", "OTHER"),
                    "text": src[pst:pen],
                    "start": pst,
                    "end": pen
                })
                last_end = pen

            fixed_clauses.append({
                "text": src[cst:cen],
                "start": cst,
                "end": cen,
                "phrases": fixed_phr
            })

        fixed_sents.append({
            "text": src[st:en],
            "start": st,
            "end": en,
            "clauses": fixed_clauses
        })

    return {"input": src, "sentences": fixed_sents}


def validate_field_result(field_input: str, fr: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    if not isinstance(fr, dict):
        return ["fieldResult is not a dict"]
    if "input" not in fr or "sentences" not in fr:
        return ["fieldResult missing keys"]
    src = field_input if field_input is not None else ""
    sentences = fr.get("sentences", [])
    sent_spans: List[Tuple[int, int]] = []
    for si, s in enumerate(sentences):
        st = s.get("start", 0)
        en = s.get("end", 0)
        txt = s.get("text", "")
        if not (isinstance(st, int) and isinstance(en, int) and 0 <= st < en <= len(src)):
            errs.append(f"Sentence {si} bad span {st}:{en}")
        else:
            if src[st:en] != txt:
                errs.append(f"Sentence {si} text mismatch at {st}:{en}")
        sent_spans.append((st, en))

        clauses = s.get("clauses", [])
        clause_spans: List[Tuple[int, int]] = []
        for ci, c in enumerate(clauses):
            cst = c.get("start", 0)
            cen = c.get("end", 0)
            ctxt = c.get("text", "")
            if not (isinstance(cst, int) and isinstance(cen, int) and st <= cst < cen <= en):
                errs.append(f"Clause {si}:{ci} bad span {cst}:{cen}")
            else:
                if src[cst:cen] != ctxt:
                    errs.append(f"Clause {si}:{ci} text mismatch at {cst}:{cen}")
            clause_spans.append((cst, cen))

            ph_spans: List[Tuple[int, int]] = []
            for pi, p in enumerate(c.get("phrases", [])):
                pst = p.get("start", 0)
                pen = p.get("end", 0)
                ptxt = p.get("text", "")
                if not (isinstance(pst, int) and isinstance(pen, int) and cst <= pst < pen <= cen):
                    errs.append(f"Phrase {si}:{ci}:{pi} bad span {pst}:{pen}")
                else:
                    if src[pst:pen] != ptxt:
                        errs.append(f"Phrase {si}:{ci}:{pi} text mismatch at {pst}:{pen}")
                ph_spans.append((pst, pen))
            if not non_overlapping(ph_spans):
                errs.append(f"Phrases overlap in clause {si}:{ci}")

        if not non_overlapping(clause_spans):
            errs.append(f"Clauses overlap in sentence {si}")

    if not non_overlapping(sent_spans):
        errs.append("Sentences overlap")
    return errs


def validate_record(record_out: Dict[str, Any], fields: List[str]) -> List[str]:
    errs: List[str] = []
    chunks = record_out.get("chunks", {})
    for field in fields:
        fr = chunks.get(field, {"input": record_out.get(field, ""), "sentences": []})
        field_input = fr.get("input", record_out.get(field, ""))
        errs += [f"[{field}] " + e for e in validate_field_result(field_input, fr)]
    return errs


# -------- dynamic splitter loading --------
def load_splitter(cfg: Dict[str, Any]):
    backend = cfg["backend"]

    if backend == "spacy":
        mod = importlib.import_module(cfg["spacy_module"])
        fn = getattr(mod, cfg["spacy_fn"])

        def splitter(text: str) -> Dict[str, Any]:
            try:
                raw = fn(text)
                if isinstance(raw, dict) and "input" in raw and "sentences" in raw:
                    fr = raw
                else:
                    fr = {"input": raw.get("text", text), "sentences": raw.get("sentences", [])}
            except Exception as e:
                fr = {"input": text, "sentences": [], "error": f"{type(e).__name__}: {e}"}
            return repair_field_result(fr)

        return splitter

    if backend == "llm":
        mod = importlib.import_module(cfg["llm_module"])
        # Optional: override LM Studio settings if exposed in module
        if cfg.get("llm_base_url") and hasattr(mod, "LMSTUDIO_BASE_URL"):
            setattr(mod, "LMSTUDIO_BASE_URL", cfg["llm_base_url"])
        if cfg.get("llm_model") and hasattr(mod, "LMSTUDIO_MODEL"):
            setattr(mod, "LMSTUDIO_MODEL", cfg["llm_model"])
        if cfg.get("llm_api_key") and hasattr(mod, "LMSTUDIO_API_KEY"):
            setattr(mod, "LMSTUDIO_API_KEY", cfg["llm_api_key"])

        fn = getattr(mod, cfg["llm_fn"])

        def splitter(text: str) -> Dict[str, Any]:
            try:
                raw = fn(text)  # LLM module should return {"text": ..., "sentences": ...} or include "error"
                fr = {"input": raw.get("text", text), "sentences": raw.get("sentences", [])}
            except Exception as e:
                fr = {"input": text, "sentences": [], "error": f"{type(e).__name__}: {e}"}
            return repair_field_result(fr)

        return splitter

    raise ValueError(f"Unknown backend {backend}")


# -------- core processing --------
def chunk_fields(item: Dict[str, Any], splitter, fields: List[str]) -> Dict[str, Any]:
    out = {
        "text": item.get("text", ""),
        "span": item.get("span", ""),
        "labels": item.get("labels", []),
        "selected_labels": item.get("selected_labels", []),
        "split": item.get("split", None),
        "chunks": {}
    }
    for field in fields:
        val = out.get(field, "")
        if isinstance(val, str) and val.strip():
            fr = splitter(val)
        else:
            fr = {"input": val or "", "sentences": []}
        out["chunks"][field] = repair_field_result(fr)
    return out


def ensure_dir(path: Optional[str]):
    if not path:
        return
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def default_paths(cfg: Dict[str, Any]) -> Tuple[str, str]:
    base_dir = cfg["output_dir"] or os.path.dirname(cfg["input_json"])
    base_name = os.path.splitext(os.path.basename(cfg["input_json"]))[0]
    final_out = cfg["final_out"] or os.path.join(base_dir, base_name + ".chunked.json")
    checkpoint = cfg["checkpoint"] or os.path.join(base_dir, base_name + ".checkpoint.jsonl")
    return final_out, checkpoint


def load_input(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint_ids(checkpoint_path: str) -> Dict[str, Dict[str, Any]]:
    done: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(checkpoint_path):
        return done
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                rid = row.get("id")
                rec = row.get("record")
                if isinstance(rid, str) and isinstance(rec, dict):
                    done[rid] = rec
            except Exception:
                continue
    return done


def append_checkpoint(checkpoint_path: str, rec_id: str, record_out: Dict[str, Any]):
    ensure_dir(checkpoint_path)
    with open(checkpoint_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": rec_id, "record": record_out}, ensure_ascii=False) + "\n")


def merge_checkpoint_to_final(input_ids: List[str],
                              checkpoint_done: Dict[str, Dict[str, Any]],
                              final_out_path: str):
    merged: Dict[str, Any] = {}
    for rid in input_ids:
        if rid in checkpoint_done:
            merged[rid] = checkpoint_done[rid]
    ensure_dir(final_out_path)
    tmp = final_out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    os.replace(tmp, final_out_path)


def run():
    cfg = CONFIG
    splitter = load_splitter(cfg)
    final_out_path, checkpoint_path = default_paths(cfg)

    data = load_input(cfg["input_json"])     # dict keyed by id
    all_ids = list(data.keys())

    # resume: load already processed ids from checkpoint
    done_map = load_checkpoint_ids(checkpoint_path)
    done_ids = set(done_map.keys())
    todo_ids = [rid for rid in all_ids if rid not in done_ids]

    print(f"Loaded {len(all_ids)} items. Already done {len(done_ids)}. To do {len(todo_ids)}.")
    if not todo_ids:
        print("Nothing to do. Writing final JSON from checkpoint.")
        merge_checkpoint_to_final(all_ids, done_map, final_out_path)
        print(f"Wrote final to {final_out_path}")
        return

    # optional probe on the next N items
    probe_n = max(0, min(cfg.get("probe", 0), len(todo_ids)))
    if probe_n > 0:
        print(f"Probe on {probe_n} new items")
        probe_errs_total = 0
        for rid in todo_ids[:probe_n]:
            record_out = chunk_fields(data[rid], splitter, cfg["fields"])
            errs = validate_record(record_out, cfg["fields"])
            if errs:
                probe_errs_total += 1
                print(f"[PROBE ERROR] {rid}")
                for e in errs:
                    print("  -", e)
            append_checkpoint(checkpoint_path, rid, record_out)
            done_map[rid] = record_out
        if probe_errs_total > 0 and cfg.get("stop_on_error", True):
            print(f"Probe found {probe_errs_total} items with errors. Stop-on-error is True. Stopping.")
            return
        # update todo after probe
        done_ids = set(done_map.keys())
        todo_ids = [rid for rid in all_ids if rid not in done_ids]
        print(f"Probe clean. Remaining {len(todo_ids)} items.")

    # full run on remaining items, append to checkpoint each time
    for rid in todo_ids:
        try:
            record_out = chunk_fields(data[rid], splitter, cfg["fields"])
        except Exception as e:
            record_out = {
                "text": data[rid].get("text",""),
                "span": data[rid].get("span",""),
                "labels": data[rid].get("labels",[]),
                "selected_labels": data[rid].get("selected_labels",[]),
                "split": data[rid].get("split"),
                "chunks": {
                    "text": {"input": data[rid].get("text",""), "sentences": [], "error": f"driver:{e}"},
                    "span": {"input": data[rid].get("span",""), "sentences": [], "error": f"driver:{e}"},
                }
            }
        errs = validate_record(record_out, cfg["fields"])
        if errs:
            print(f"[WARN] validation issues in {rid} (continuing):")
            for e in errs:
                print("  -", e)
        append_checkpoint(checkpoint_path, rid, record_out)
        # IMPORTANT: update done_map so final merge includes these items
        done_map[rid] = record_out

    # final merge
    merge_checkpoint_to_final(all_ids, done_map, final_out_path)
    print(f"Done. Wrote final to {final_out_path}")
    print(f"Checkpoint at {checkpoint_path}")

if __name__ == "__main__":
    run()
