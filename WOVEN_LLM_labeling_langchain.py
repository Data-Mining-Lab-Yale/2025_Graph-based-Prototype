import os, re, json, time, hashlib
from typing import Dict, Any, Optional
import pandas as pd
from tqdm import tqdm

# ====== CONFIG ======
# CASE_TYPE_FILTER = "Clinical_Question"
CASE_TYPE_FILTER = "Statement_Billing_or_Insurance_Question"
INPUT_PATH = f"Data/WOVEN/split_by_case_type/{CASE_TYPE_FILTER}.jsonl"
TEXT_COL = "Pt. Case Description"

OUT_DIR = "Data/WOVEN/auto_label_outputs"
os.makedirs(OUT_DIR, exist_ok=True)
STEM = os.path.splitext(os.path.basename(INPUT_PATH))[0]
OUT_JSONL = os.path.join(OUT_DIR, f"{STEM}_autolabeled.jsonl")
OUT_CSV   = os.path.join(OUT_DIR, f"{STEM}_autolabeled.csv")

# Relabel policy: "overwrite_all" | "when_default" | "missing_only"
RELABEL_MODE = "overwrite_all"
VERBOSE = True

# Default pattern from your earlier steps
DEFAULT_PATTERN = {
    "patient-to-provider": 0,
    "provider-to-patient": 0,
    "provider-to-provider": 1,
}

# Telephone flag handling
TELEPHONE_MODE = "keep"  # "keep" or "heuristic"
TELEPHONE_REGEX = re.compile(r"\b(phone|telephone|voicemail|vm|called|call(?:ed)?)\b", re.I)

# ====== MODEL PROVIDER ======
# Choose: "lmstudio" (free local) or "openai" (paid)
PROVIDER = "lmstudio"

# LM Studio (OpenAI compatible)
LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
LMSTUDIO_MODEL_ID = "phi-3-mini-4k-instruct"   # pick one from GET /v1/models

# OpenAI (paid) if you want to compare quality
# export OPENAI_API_KEY=...
OPENAI_MODEL_ID = "gpt-4o-mini"

# ====== Prompt ======
PROMPT_SYSTEM = """You label short clinical communication notes in an EMR.

Goal: Identify who is speaking to whom inside the note (directionality), not just who the note is about.

Return ONLY a compact JSON with integer 0/1 for these keys:
{
  "is_thread": 0 or 1,
  "patient-to-provider": 0 or 1,
  "provider-to-patient": 0 or 1,
  "provider-to-provider": 0 or 1,
  "is_content": 0 or 1
}

Core principle — narrator vs agent:
- The EMR note author is a provider/staff writing for other staff.
- Phrases like "pt called ..." usually mean the provider is narrating the patient's call to the clinic. That narration counts as provider-to-provider.
- Mark patient-to-provider only when the patient's message content itself is present as a message to the clinic.
- Mark provider-to-patient when the note contains content addressed to the patient (advice, instructions, explanations).

Cues:
- provider-to-provider: "pt called ...", "FYI to Dr X", "routed to MA", "please review", "pharmacy stated", "prior auth expired".
- patient-to-provider: "portal message from patient: '...'", "patient wrote: '...'", clear patient-authored message content.
- provider-to-patient: "RN advised ...", "called patient and explained ...", "left detailed instructions".

Threads:
- If a record contains an original message plus a reply, set is_thread = 1 and mark all directions present.

is_content:
- 0 for pure contact status only: "called at 3pm, no answer", "LVM", "wrong number", "updated phone".
- 1 if any meaningful content exists: symptoms, questions, clinical advice, assessment, plan, next steps.

Ambiguity rule:
- If directionality cannot be determined with reasonable confidence, set all three roles to 1.

Output:
- Return ONLY the JSON object. Values must be 0 or 1. No explanations.
"""

PROMPT_USER_TMPL = "Text:\n{note}\n\nReturn only the JSON object."

# ====== LangChain setup ======
from langchain.prompts import ChatPromptTemplate
if PROVIDER == "lmstudio":
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        base_url=LMSTUDIO_BASE_URL,
        api_key="not-needed",
        model=LMSTUDIO_MODEL_ID,
        temperature=0.0,
    )
elif PROVIDER == "openai":
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=OPENAI_MODEL_ID,
        temperature=0.0,
    )
else:
    raise ValueError("PROVIDER must be 'lmstudio' or 'openai'.")

prompt = ChatPromptTemplate.from_messages([
    ("system", PROMPT_SYSTEM),
    ("user", "{user_msg}")
])
chain = prompt | llm

# ====== Helpers ======
def read_any(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(os.path.abspath(path))
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
    raise ValueError("Unsupported input extension.")

def stable_id(text: str) -> str:
    h = hashlib.sha256()
    h.update((text or "").encode("utf-8"))
    return h.hexdigest()[:16]

def extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        block = m.group(0)
        for _ in range(2):
            try:
                return json.loads(block)
            except Exception:
                block = re.sub(r",\s*}", "}", block)
                block = re.sub(r",\s*]", "]", block)
    return None

def to01(v) -> int:
    try:
        return 1 if int(v) == 1 else 0
    except Exception:
        return 1 if str(v).strip() == "1" else 0

def label_one(text: str) -> Dict[str, Any]:
    user = PROMPT_USER_TMPL.format(note=text or "")
    resp = chain.invoke({"user_msg": user})
    raw = getattr(resp, "content", str(resp))
    out = {
        "is_thread": 0,
        "patient-to-provider": 1,
        "provider-to-patient": 1,
        "provider-to-provider": 1,
        "is_content": 0,
        "_raw": raw[:4000],
    }
    parsed = extract_json_obj(raw)
    if isinstance(parsed, dict):
        out.update({
            "is_thread": to01(parsed.get("is_thread", 0)),
            "patient-to-provider": to01(parsed.get("patient-to-provider", 0)),
            "provider-to-patient": to01(parsed.get("provider-to-patient", 0)),
            "provider-to-provider": to01(parsed.get("provider-to-provider", 0)),
            "is_content": to01(parsed.get("is_content", 0)),
        })
        if (out["patient-to-provider"] + out["provider-to-patient"] + out["provider-to-provider"]) == 0:
            out["patient-to-provider"]  = 1
            out["provider-to-patient"]  = 1
            out["provider-to-provider"] = 1
    return out

def derive_is_multiple(rec: Dict[str, Any]) -> int:
    s = int(rec.get("patient-to-provider",0)) + int(rec.get("provider-to-patient",0)) + int(rec.get("provider-to-provider",0))
    return 1 if s > 1 else 0

def maybe_set_telephone(row: dict, existing_val: int) -> int:
    if TELEPHONE_MODE == "keep":
        return int(existing_val)
    if TELEPHONE_MODE == "heuristic":
        text = str(row.get(TEXT_COL,"") or "")
        if existing_val in (1, "1"):
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

def append_jsonl(path: str, record: Dict[str, Any]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ====== Main ======
def main():
    print("=== Step 1: Load data ===")
    print(f"Input: {os.path.abspath(INPUT_PATH)}")
    df = read_any(INPUT_PATH)
    print(f"Rows: {len(df)}")
    if TEXT_COL not in df.columns:
        raise KeyError(f'Missing column: "{TEXT_COL}"')

    # ensure columns exist
    for c in ["patient-to-provider","provider-to-patient","provider-to-provider",
              "is_telephone_note","is_content","is_thread"]:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    print("\n=== Step 2: Labeling ===")
    rows_for_csv = []
    labeled_via_llm = 0
    kept_existing = 0

    for i, row in tqdm(df.iterrows(), total=len(df), unit="row"):
        text = str(row.get(TEXT_COL, "") or "")
        rid  = stable_id(text)

        ptp  = int(row.get("patient-to-provider", 0))
        prtp = int(row.get("provider-to-patient", 0))
        prpr = int(row.get("provider-to-provider", 0))
        role_sum_existing = ptp + prtp + prpr

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
            labeled = label_one(text)
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

        # derive and set is_multiple
        labeled["is_multiple"] = derive_is_multiple(labeled)

        # telephone flag pass through or heuristic
        tel_existing = int(row.get("is_telephone_note", 0))
        labeled["is_telephone_note"] = maybe_set_telephone(row, tel_existing)

        rec = {**row.to_dict(), **labeled, "_rid": rid}
        append_jsonl(OUT_JSONL, rec)
        rows_for_csv.append(rec)

        if VERBOSE and i < 5:
            before = (ptp, prtp, prpr)
            after  = (labeled["patient-to-provider"], labeled["provider-to-patient"], labeled["provider-to-provider"])
            print(f"[debug] row {i}: decision={reason}, roles_before={before}, roles_after={after}")

        if i % 20 == 0 and i > 0:
            time.sleep(0.1)

    out_df = pd.DataFrame(rows_for_csv)
    out_df.to_csv(OUT_CSV, index=False)

    print("\n=== Step 3: Summary ===")
    print(f"Saved JSONL: {os.path.abspath(OUT_JSONL)}")
    print(f"Saved CSV  : {os.path.abspath(OUT_CSV)}")
    print(f"Labeled rows via LLM: {labeled_via_llm}")
    print(f"Kept existing rows : {kept_existing}")
    for col in ["is_thread","is_content","is_telephone_note",
                "patient-to-provider","provider-to-patient","provider-to-provider","is_multiple"]:
        if col in out_df.columns:
            total = int(pd.to_numeric(out_df[col], errors="coerce").fillna(0).astype(int).sum())
            print(f"{col}: {total}")

if __name__ == "__main__":
    main()
