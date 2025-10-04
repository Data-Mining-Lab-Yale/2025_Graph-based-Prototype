# chunk_driver_config.py
# Run:
#   python chunk_driver_config.py                 # uses CONFIG["default"]
#   python chunk_driver_config.py --task llm5     # uses TASKS["llm5"]
#   python chunk_driver_config.py --task spacy5   # uses TASKS["spacy5"]

import os, json, importlib, argparse
from typing import Any, Dict, List, Tuple, Optional

# ========= USER CONFIG =========
CONFIG = {
    "default": {
        # paths
        "input_json": f"toysets_out/goal_toyset_raw.json",
        "output_dir": f"Phrase_out",           # folder for outputs, None means same folder as input
        "out": None,                       # full output path, overrides output_dir if set
        "out_probe": None,                 # probe output path, overrides output_dir if set

        # what to process
        "fields": ["text", "span"],        # choose from ["text", "span"]
        "probe": 5,                        # process first N items, validate, then continue if clean
        "backend": "llm",                  # "spacy" or "llm"

        # spaCy backend location
        "spacy_module": "Decision_Structure_1_chucking_v1",
        "spacy_fn": "split_text_into_clauses_and_phrases",

        # LLM backend location and settings
        "llm_module": "Decision_Structure_1_chucking_local_llm",
        "llm_fn": "lmstudio_chunk",
        "llm_base_url": "http://127.0.0.1:1234/v1",
        "llm_model": "openai/gpt-oss-20b",
        "llm_api_key": "lm-studio"
    }
}

# Presets you can select with --task
# TASKS = {
#     # probe 5 with LM Studio
#     "llm5": {
#         "backend": "llm",
#         "probe": 5
#     },
#     # probe 5 with spaCy
#     "spacy5": {
#         "backend": "spacy",
#         "probe": 5
#     }
# }
TASKS = {
    "llm_all":   {"backend": "llm",   "probe": 0},
    "spacy_all": {"backend": "spacy", "probe": 0},
    # keep your earlier presets too
    "llm5": {"backend": "llm", "probe": 5},
    "spacy5": {"backend": "spacy", "probe": 5},
}
# ========= END CONFIG =========


# -------- validation --------
def non_overlapping(spans: List[Tuple[int, int]]) -> bool:
    spans = sorted(spans)
    for i in range(1, len(spans)):
        if spans[i][0] < spans[i - 1][1]:
            return False
    return True

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
            raw = fn(text)
            if isinstance(raw, dict) and "input" in raw and "sentences" in raw:
                return raw
            return {"input": raw.get("text", text), "sentences": raw.get("sentences", [])}
        return splitter

    if backend == "llm":
        mod = importlib.import_module(cfg["llm_module"])
        if cfg.get("llm_base_url") and hasattr(mod, "LMSTUDIO_BASE_URL"):
            setattr(mod, "LMSTUDIO_BASE_URL", cfg["llm_base_url"])
        if cfg.get("llm_model") and hasattr(mod, "LMSTUDIO_MODEL"):
            setattr(mod, "LMSTUDIO_MODEL", cfg["llm_model"])
        if cfg.get("llm_api_key") and hasattr(mod, "LMSTUDIO_API_KEY"):
            setattr(mod, "LMSTUDIO_API_KEY", cfg["llm_api_key"])
        fn = getattr(mod, cfg["llm_fn"])
        def splitter(text: str) -> Dict[str, Any]:
            raw = fn(text)
            return {"input": raw.get("text", text), "sentences": raw.get("sentences", [])}
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
        out["chunks"][field] = fr
    return out

def write_json(path: str, obj: Any):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -------- config helpers --------
def merged_config(task_name: Optional[str]) -> Dict[str, Any]:
    base = dict(CONFIG["default"])
    if task_name:
        if task_name not in TASKS:
            raise ValueError(f"Unknown task '{task_name}'. Choices: {list(TASKS.keys())}")
        base.update(TASKS[task_name])
    return base

def compute_out_paths(cfg: Dict[str, Any]) -> Tuple[str, str]:
    in_path = cfg["input_json"]
    base_name = os.path.splitext(os.path.basename(in_path))[0]
    base_dir = cfg["output_dir"] or os.path.dirname(in_path)
    out_probe_path = cfg["out"] if cfg["out_probe"] is None else cfg["out_probe"]
    out_full_path = cfg["out"]
    if out_probe_path is None:
        out_probe_path = os.path.join(base_dir, base_name + ".probe.chunked.json")
    if out_full_path is None:
        out_full_path = os.path.join(base_dir, base_name + ".chunked.json")
    return out_probe_path, out_full_path


# -------- main --------
def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--task", default=None, help="Preset name in TASKS, for example llm5 or spacy5")
    args, _ = ap.parse_known_args()

    cfg = merged_config(args.task)
    splitter = load_splitter(cfg)

    with open(cfg["input_json"], "r", encoding="utf-8") as f:
        data = json.load(f)   # dict keyed by id

    ids = list(data.keys())
    probe_n = max(0, min(cfg["probe"], len(ids)))
    out_probe_path, out_full_path = compute_out_paths(cfg)

    print(f"Probe pass on {probe_n} items. Backend={cfg['backend']}. Fields={cfg['fields']}")
    probe_out: Dict[str, Any] = {}
    all_errs: Dict[str, List[str]] = {}
    for k in ids[:probe_n]:
        probe_out[k] = chunk_fields(data[k], splitter, cfg["fields"])
        errs = validate_record(probe_out[k], cfg["fields"])
        if errs:
            all_errs[k] = errs

    write_json(out_probe_path, probe_out)
    print(f"Wrote probe results to {out_probe_path}")

    if all_errs:
        print("Probe found validation issues. Stopping before full run.")
        for k, errs in all_errs.items():
            print(f"Item {k}:")
            for e in errs:
                print("  -", e)
        return

    print("Probe clean. Continuing with full run.")
    full_out: Dict[str, Any] = {**probe_out}
    for k in ids[probe_n:]:
        full_out[k] = chunk_fields(data[k], splitter, cfg["fields"])

    write_json(out_full_path, full_out)
    print(f"Wrote {len(full_out)} items to {out_full_path}")

if __name__ == "__main__":
    main()
