# WOVEN_LLM_labeling.py  (LM Studio, recheck + overwrite capable)
import os, re, json, time, hashlib
from typing import Dict, Any, Optional
import pandas as pd
import requests
from tqdm import tqdm

# ---------------- CONFIG ----------------
CASE_TYPE_FILTER = "Clinical_Question"  # e.g., "Clinical_Question", "Medication"
INPUT_PATH = f"Data/WOVEN/split_by_case_type/{CASE_TYPE_FILTER}.jsonl"
TEXT_COL = "Pt. Case Description"

OUT_DIR = "Data/WOVEN/auto_label_outputs"
os.makedirs(OUT_DIR, exist_ok=True)
STEM = os.path.splitext(os.path.basename(INPUT_PATH))[0]
OUT_JSONL = os.path.join(OUT_DIR, f"{STEM}_autolabeled.jsonl")
OUT_CSV   = os.path.join(OUT_DIR, f"{STEM}_autolabeled.csv")

# Recheck mode: "overwrite_all" | "when_default" | "missing_only"
RELABEL_MODE = "overwrite_all"
VERBOSE = True

# Known default pattern from your earlier steps
DEFAULT_PATTERN = {
    "patient-to-provider": 0,
    "provider-to-patient": 0,
    "provider-to-provider": 1,
}

# Optional: start fresh each run so outputs don’t duplicate rows
CLEAR_PREVIOUS_OUTPUT = False
if CLEAR_PREVIOUS_OUTPUT:
    for p in (OUT_JSONL, OUT_CSV):
        try:
            if os.path.exists(p): os.remove(p)
        except Exception:
            pass

# Preserve telephone flag as-is. You can switch to "heuristic" to auto-set if missing.
TELEPHONE_MODE = "keep"  # "keep" | "heuristic"
TELEPHONE_REGEX = re.compile(r"\b(phone|telephone|voicemail|vm|called|call(ed)?|lvm)\b", re.I)

# ---------------- LM Studio API ----------------
LMSTUDIO_BASE = "http://localhost:1234/v1"
LMSTUDIO_MODELS = f"{LMSTUDIO_BASE}/models"
LMSTUDIO_CHAT  = f"{LMSTUDIO_BASE}/chat/completions"

# Pick a model from GET /v1/models (your list showed these IDs)
LMSTUDIO_MODEL = "openai/gpt-oss-20b"  # or: "mistral-7b-instruct-v0.3", "llama-3.2-3b-instruct"

def check_lmstudio_server():
    r = requests.get(LMSTUDIO_MODELS, timeout=5)
    r.raise_for_status()
    return r.json()

def ensure_lmstudio_model(model_id: str, models_json: dict):
    ids = [m.get("id","") for m in models_json.get("data",[])]
    if model_id not in ids:
        raise RuntimeError(
            f"Model '{model_id}' not found.\n"
            f"Available: {ids}\n"
            "Open LM Studio → Models → Download a model, then use its 'id' here."
        )

def probe_lmstudio_model(model_id: str) -> str:
    payload = {
        "model": model_id,
        "messages": [
            {"role":"system","content":"Return only JSON."},
            {"role":"user","content":"{\"ok\":1}"}
        ],
        "temperature": 0
    }
    r = requests.post(LMSTUDIO_CHAT, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

def call_lmstudio(prompt_system: str, prompt_user: str) -> str:
    payload = {
        "model": LMSTUDIO_MODEL,
        "messages": [
            {"role":"system","content": prompt_system},
            {"role":"user","content": prompt_user}
        ],
        "temperature": 0,
    }
    r = requests.post(LMSTUDIO_CHAT, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

# ---------------- Prompts ----------------
# PROMPT_SYSTEM = """You label short clinical communication notes.

# Return ONLY a compact JSON with integer 0/1 for these keys:
# {
#   "is_thread": 0 or 1,                // 1 if the note contains multiple messages in one thread
#   "patient-to-provider": 0 or 1,
#   "provider-to-patient": 0 or 1,
#   "provider-to-provider": 0 or 1,
#   "is_content": 0 or 1                // 1 if there is substantive communication content
# }

# Rules:
# - If you cannot tell who talks to whom with reasonable confidence, set ALL THREE role flags to 1.
# - is_content = 0 examples: "called at 3pm", "left voicemail", "attempted to reach provider", "contact info updated".
# - is_content = 1 examples: specific questions, advice, symptoms, decisions, findings, next steps.
# """
PROMPT_SYSTEM = """You are labeling short clinical communication notes from patient-provider workflows.
Your job is to output ONLY a compact JSON object with integer 0/1 for these exact keys:
{
  "is_thread": 0 or 1,                // 1 if the note contains multiple messages (reply + original) in one record
  "patient-to-provider": 0 or 1,      // at least one message authored by a patient addressed to a provider
  "provider-to-patient": 0 or 1,      // at least one message authored by a provider addressed to the patient
  "provider-to-provider": 0 or 1,     // at least one message authored by a provider addressed to another provider/staff
  "is_content": 0 or 1                // 1 if substantive clinical/communication content appears (beyond pure contact status)
}

DEFINITIONS & CUES (use multiple weak cues to decide; prefer conservative decisions):
- "patient-to-provider" cues:
  * Patient speaks or is quoted asking, reporting symptoms, information, or requests to clinic.
  * Phrases like "patient said/asked/reports", "pt called to ask...", first-person lines clearly from patient context.
  * Messages containing patient questions, concerns, symptom narratives, scheduling requests initiated by patient.
- "provider-to-patient" cues:
  * Provider/staff communicates to patient: advice, instructions, triage outcomes, appointment confirmations, callbacks.
  * Phrases like "RN advised...", "MA informed patient...", "provider called patient and explained...", "left detailed instructions".
- "provider-to-provider" cues:
  * Internal routing among staff/providers: handoffs, FYIs, orders, referrals, chart notes for colleagues, "sent to Dr X", "messaged MA".
  * No direct patient addressing.
  * Typical EMR phrases: "route to", "FYI to PCP", "please review", "forwarded to nurse pool".
- If a single record contains a thread (original + reply), set "is_thread" = 1 and mark ALL roles present in that thread.

"is_content":
- Set 0 when the note is purely administrative contact status without substance: "called at 3pm", "left voicemail", "no answer", "wrong number", "updated phone", "LVM", "attempted to reach", "scheduled for X" with no clinical info.
- Set 1 if there is any meaningful content: symptoms, questions, clinical advice, medication changes, assessment, plan, decisions, next steps.

AMBIGUITY & RULES:
- If directionality cannot be determined with reasonable confidence, set ALL THREE role flags to 1 (union) to avoid missing cases.
- Multiple roles can be 1 in the same record if the thread includes messages in different directions.
- Avoid hallucinating content: if a phrase is unclear and could be administrative only, prefer is_content = 0.

OUTPUT FORMAT:
- Return ONLY the JSON object (no prose, no markdown).
- Values must be integers 0 or 1.
- Do not include explanations.

EXAMPLES (illustrative; do not copy text):
1) "Pt called asking if dizziness could be from new med. RN returned call and advised to take with food."
   -> is_thread=1, pt→pr=1, pr→pt=1, pr→pr=0, is_content=1

2) "Called patient 2x, no answer. LVM to call clinic back."
   -> is_thread=0, pt→pr=0, pr→pt=1 (provider left message), pr→pr=0, is_content=0

3) "FYI to Dr. Lee: pharmacy says prior auth expired."
   -> is_thread=0, pt→pr=0, pr→pt=0, pr→pr=1, is_content=1

4) "Left voicemail. Wrong number."
   -> is_thread=0, pt→pr=0, pr→pt=1, pr→pr=0, is_content=0
"""




PROMPT_USER_TEMPLATE = "Text:\n{note}\n\nReturn only the JSON object."

# ---------------- IO helpers ----------------
def read_any(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {os.path.abspath(path)}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        return pd.read_json(path, lines=True)
    if ext == ".json":
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return pd.json_normalize(data, sep=".")
            raise ValueError("JSON is not JSONL or an array.")
    if ext == ".csv":
        return pd.read_csv(path)
    raise ValueError("INPUT_PATH must be .jsonl, .json, or .csv")

def stable_id(text: str) -> str:
    h = hashlib.sha256()
    h.update((text or "").encode("utf-8"))
    return h.hexdigest()[:16]

def load_resume_map(out_jsonl: str):
    if not os.path.exists(out_jsonl):
        return {}
    m = {}
    with open(out_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rid = obj.get("_rid")
                if rid:
                    m[rid] = obj
            except Exception:
                pass
    return m

def append_jsonl(path: str, record: Dict[str, Any]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ---------------- JSON extraction ----------------
def extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        chunk = m.group(0)
        try:
            return json.loads(chunk)
        except Exception:
            chunk2 = re.sub(r",\s*}", "}", chunk)
            chunk2 = re.sub(r",\s*]", "]", chunk2)
            try:
                return json.loads(chunk2)
            except Exception:
                return None
    return None

# ---------------- LLM labeling ----------------
def label_with_llm(note: str) -> Dict[str, Any]:
    raw = call_lmstudio(PROMPT_SYSTEM, PROMPT_USER_TEMPLATE.format(note=note or ""))

    # defaults if parsing fails: your rule → set all roles to 1
    out = {
        "is_thread": 0,
        "patient-to-provider": 1,
        "provider-to-patient": 1,
        "provider-to-provider": 1,
        "is_content": 0,
        "_raw": raw[:4000]
    }
    parsed = extract_json_obj(raw)
    if isinstance(parsed, dict):
        def to01(v):
            try: return 1 if int(v) == 1 else 0
            except Exception: return 1 if str(v).strip() == "1" else 0
        out.update({
            "is_thread":            to01(parsed.get("is_thread", 0)),
            "patient-to-provider":  to01(parsed.get("patient-to-provider", 0)),
            "provider-to-patient":  to01(parsed.get("provider-to-patient", 0)),
            "provider-to-provider": to01(parsed.get("provider-to-provider", 0)),
            "is_content":           to01(parsed.get("is_content", 0)),
        })
        # if model gave all roles 0, set all to 1
        if (out["patient-to-provider"] + out["provider-to-patient"] + out["provider-to-provider"]) == 0:
            out["patient-to-provider"]  = 1
            out["provider-to-patient"]  = 1
            out["provider-to-provider"] = 1
    return out

def derive_is_multiple(rec: Dict[str, Any]) -> int:
    role_sum = int(rec.get("patient-to-provider",0)) + int(rec.get("provider-to-patient",0)) + int(rec.get("provider-to-provider",0))
    return 1 if role_sum > 1 else 0

def maybe_set_telephone(row: dict, existing_val: int) -> int:
    if TELEPHONE_MODE == "keep":
        return int(existing_val)
    if TELEPHONE_MODE == "heuristic":
        text = str(row.get(TEXT_COL,"") or "")
        if existing_val in (1, "1"):  # already set
            return 1
        return 1 if TELEPHONE_REGEX.search(text) else 0
    return int(existing_val)

def is_default_roles(row) -> bool:
    try:
        return int(row.get("patient-to-provider", 0)) == DEFAULT_PATTERN["patient-to-provider"] \
           and int(row.get("provider-to-patient", 0)) == DEFAULT_PATTERN["provider-to-patient"] \
           and int(row.get("provider-to-provider", 0)) == DEFAULT_PATTERN["provider-to-provider"]
    except Exception:
        return False

# ---------------- Main ----------------
def main():
    print("=== Step 0: Checking LM Studio server and model ===")
    models_json = check_lmstudio_server()
    print("LM Studio server OK at http://localhost:1234")
    ensure_lmstudio_model(LMSTUDIO_MODEL, models_json)
    probe = probe_lmstudio_model(LMSTUDIO_MODEL)
    print(f"Model '{LMSTUDIO_MODEL}' is available. Probe: {probe[:80]!r}")

    print("\n=== Step 1: Loading input data ===")
    print(f"Input file: {os.path.abspath(INPUT_PATH)}")
    df = read_any(INPUT_PATH)
    print(f"Rows loaded: {len(df)}")
    if TEXT_COL not in df.columns:
        raise KeyError(f'Missing column: \"{TEXT_COL}\"')
    print(f'Using text column: \"{TEXT_COL}\"')

    print("\n=== Step 2: Outputs & resume map ===")
    print(f"JSONL out: {os.path.abspath(OUT_JSONL)}")
    print(f"CSV   out: {os.path.abspath(OUT_CSV)}")
    resume = load_resume_map(OUT_JSONL)
    print(f"Resume rows found: {len(resume)}")

    # ensure flag columns exist so we can read/keep them
    for c in ["patient-to-provider","provider-to-patient","provider-to-provider",
              "is_telephone_note","is_content","is_thread"]:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    print("\n=== Step 3: Labeling (recheck mode) ===")
    rows_for_csv = []
    labeled_via_llm = 0
    kept_existing   = 0

    for i, row in tqdm(df.iterrows(), total=len(df), unit="row"):
        text = str(row.get(TEXT_COL, "") or "")
        rid  = stable_id(text)

        # read existing flags
        ptp  = int(row.get("patient-to-provider", 0))
        prtp = int(row.get("provider-to-patient", 0))
        prpr = int(row.get("provider-to-provider", 0))
        role_sum_existing = ptp + prtp + prpr

        # decide recheck
        if RELABEL_MODE == "overwrite_all":
            should_label = True
            reason = "overwrite_all"
        elif RELABEL_MODE == "when_default":
            if role_sum_existing == 0:
                should_label = True; reason = "missing_roles"
            elif is_default_roles(row):
                should_label = True; reason = "default_pattern"
            else:
                should_label = False; reason = "non_default_existing"
        elif RELABEL_MODE == "missing_only":
            should_label = (role_sum_existing == 0)
            reason = "missing_roles" if should_label else "has_roles"
        else:
            should_label = True; reason = "overwrite_all_fallback"

        if should_label:
            labeled = label_with_llm(text)
            labeled_via_llm += 1
        else:
            labeled = {
                "is_thread": int(row.get("is_thread", 0)),
                "patient-to-provider": ptp,
                "provider-to-patient": prtp,
                "provider-to-provider": prpr,
                "is_content": int(row.get("is_content", 0)),
            }
            kept_existing += 1

        # derive is_multiple from roles
        labeled["is_multiple"] = derive_is_multiple(labeled)

        # pass through or heuristic for telephone flag
        tel_existing = int(row.get("is_telephone_note", 0))
        labeled["is_telephone_note"] = maybe_set_telephone(row, tel_existing)

        # persist
        rec = {**row.to_dict(), **labeled, "_rid": rid}
        append_jsonl(OUT_JSONL, rec)
        rows_for_csv.append(rec)

        if VERBOSE and i < 5:
            before = (ptp, prtp, prpr)
            after  = (labeled["patient-to-provider"], labeled["provider-to-patient"], labeled["provider-to-provider"])
            print(f"[debug] row {i}: decision={reason}, roles_before={before}, roles_after={after}")

        # light throttle
        if i % 20 == 0 and i > 0:
            time.sleep(0.15)

    out_df = pd.DataFrame(rows_for_csv)
    out_df.to_csv(OUT_CSV, index=False)

    print("\n=== Step 4: Summary ===")
    print(f"Saved JSONL: {os.path.abspath(OUT_JSONL)}")
    print(f"Saved CSV  : {os.path.abspath(OUT_CSV)}")
    print(f"Labeled rows via LLM: {labeled_via_llm}")
    print(f"Kept existing rows : {kept_existing}")
    if VERBOSE:
        for col in ["is_thread","is_content","is_telephone_note",
                    "patient-to-provider","provider-to-patient","provider-to-provider","is_multiple"]:
            if col in out_df.columns:
                total = int(pd.to_numeric(out_df[col], errors="coerce").fillna(0).astype(int).sum())
                print(f"{col}: {total}")

if __name__ == "__main__":
    main()
