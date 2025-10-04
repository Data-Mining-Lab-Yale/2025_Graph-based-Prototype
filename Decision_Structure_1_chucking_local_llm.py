# ==========================================
# Local LLM clause + phrase chunker (LM Studio)
# - Uses LM Studio OpenAI-compatible API
# - Returns clauses + phrase chunks with indices
# - Strong JSON extractor handles code fences and extra text
#
# Setup:
#   pip install openai
#   In LM Studio: start server, load "openai/gpt-oss-20b"
#   Default endpoint: http://127.0.0.1:1234/v1
# ==========================================

import json
import re
from typing import Dict, Any, List, Tuple, Optional
from openai import OpenAI
from openai.types.chat import ChatCompletion

# ---------------- config ----------------
LMSTUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
LMSTUDIO_API_KEY = "lm-studio"   # LM Studio accepts any non-empty string
LMSTUDIO_MODEL   = "openai/gpt-oss-20b"

client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key=LMSTUDIO_API_KEY)


# --------------- prompt builder ---------------

def build_prompt(text: str) -> str:
    examples = [
        {
            "text": "we ran out of time yesterday afternoon",
            "sentences": [
                {
                    "sentence_text": "we ran out of time yesterday afternoon",
                    "clauses": [
                        {
                            "clause_text": "we ran out of time yesterday afternoon",
                            "start": 0,
                            "end": 39,
                            "phrases": [
                                {"type": "NP",     "text": "we",                  "start": 0,  "end": 2},
                                {"type": "VP_PHV", "text": "ran out of time",     "start": 3,  "end": 18},
                                {"type": "ADVP",   "text": "yesterday afternoon", "start": 19, "end": 39}
                            ]
                        }
                    ]
                }
            ]
        },
        {
            "text": "The good doctor is swamped in a emails and she wants to make sure he sees it",
            "sentences": [
                {
                    "sentence_text": "The good doctor is swamped in a emails and she wants to make sure he sees it",
                    "clauses": [
                        {
                            "clause_text": "The good doctor is swamped in a emails",
                            "start": 0,
                            "end": 41,
                            "phrases": [
                                {"type": "NP",     "text": "The good doctor", "start": 0,  "end": 15},
                                {"type": "VP_COP", "text": "is swamped",      "start": 16, "end": 26},
                                {"type": "PP",     "text": "in a emails",     "start": 27, "end": 39}
                            ]
                        },
                        {
                            "clause_text": "she wants to make sure",
                            "start": 46,
                            "end": 69,
                            "phrases": [
                                {"type": "NP",   "text": "she",           "start": 46, "end": 49},
                                {"type": "VP",   "text": "wants",         "start": 50, "end": 55},
                                {"type": "INF",  "text": "to make sure",  "start": 56, "end": 69}
                            ]
                        },
                        {
                            "clause_text": "he sees it",
                            "start": 70,
                            "end": 81,
                            "phrases": [
                                {"type": "NP", "text": "he",  "start": 70, "end": 72},
                                {"type": "VP", "text": "sees","start": 73, "end": 77},
                                {"type": "NP", "text": "it",  "start": 78, "end": 80}
                            ]
                        }
                    ]
                }
            ]
        }
    ]

    schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "sentences": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "sentence_text": {"type": "string"},
                        "clauses": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "clause_text": {"type": "string"},
                                    "start": {"type": "integer"},
                                    "end": {"type": "integer"},
                                    "phrases": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "type": {"type": "string"},
                                                "text": {"type": "string"},
                                                "start": {"type": "integer"},
                                                "end": {"type": "integer"}
                                            },
                                            "required": ["type", "text", "start", "end"]
                                        }
                                    }
                                },
                                "required": ["clause_text", "start", "end", "phrases"]
                            }
                        }
                    },
                    "required": ["sentence_text", "clauses"]
                }
            }
        },
        "required": ["text", "sentences"]
    }

    rules = """
You are an NLP chunker. For the given English input string:
1) Split into clauses, then split each clause into phrase chunks.
2) Allowed chunk types: NP, VP, VP_PHV, VP_COP, PP, ADVP, ADJP, INF, DISC, PRT, OTHER.
3) Phrasal verbs are one VP_PHV such as "ran out of time". Copula plus predicate is VP_COP such as "is swamped".
4) A PP includes the preposition and its object. Do not duplicate the same characters across multiple chunks in the same clause.
5) Use exact substrings of the original text. Keep punctuation and spacing.
6) Provide character indices [start, end) relative to the original text. Indices must match the exact substring.
7) Spans do not overlap inside a clause. Clauses do not overlap and must stay in order.
8) Keep apology or interjection fragments like "SO sorry" as DISC.
9) If the input is truncated, still output what is present without inventing words.
Return a single JSON object only and follow the schema shown in the examples.
"""

    user = {
        "task": "Chunk the following text",
        "schema_hint": schema,
        "examples": examples,
        "text": text
    }
    return f"{rules}\n\nINPUT:\n{json.dumps(user, ensure_ascii=False)}"


# --------------- JSON extraction and validation ---------------

def strip_code_fences(s: str) -> str:
    # Remove ```json ... ``` or ``` ... ```
    s = re.sub(r"^```(?:json)?\s*", "", s.strip(), flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s

def find_balanced_json_payload(s: str) -> Optional[str]:
    """
    Return the first balanced top-level JSON object or array.
    Handles braces inside strings and escapes.
    """
    s = strip_code_fences(s)
    # find first opening { or [
    start_candidates = [i for i, ch in enumerate(s) if ch in "{["]
    for start in start_candidates:
        stack = []
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch in "{[":
                    stack.append("}" if ch == "{" else "]")
                elif ch in "}]":
                    if not stack or ch != stack[-1]:
                        break
                    stack.pop()
                    if not stack:
                        # candidate complete
                        return s[start:i+1]
        # try next start if this one failed
    return None

def extract_json(maybe_json: str) -> Dict[str, Any]:
    # First, try direct parse
    try:
        return json.loads(strip_code_fences(maybe_json))
    except Exception:
        pass
    # Next, try balanced scan
    payload = find_balanced_json_payload(maybe_json)
    if payload is not None:
        try:
            return json.loads(payload)
        except Exception as e:
            raise ValueError(f"Found JSON-like payload but failed to parse: {e}\nPayload:\n{payload}") from e
    # As a last resort, search all balanced candidates and pick the longest valid
    candidates = re.findall(r"\{.*\}|\[.*\]", strip_code_fences(maybe_json), flags=re.DOTALL)
    best: Optional[Dict[str, Any]] = None
    for c in sorted(candidates, key=len, reverse=True):
        try:
            best = json.loads(c)
            break
        except Exception:
            continue
    if best is not None:
        return best
    raise ValueError("Could not extract valid JSON from model output")


def non_overlapping(spans: List[Tuple[int, int]]) -> bool:
    spans = sorted(spans)
    for i in range(1, len(spans)):
        if spans[i][0] < spans[i-1][1]:
            return False
    return True

def validate_result(src: str, result: Dict[str, Any]) -> List[str]:
    errs = []
    if "sentences" not in result:
        return ["Missing 'sentences'"]
    for s in result["sentences"]:
        if "clauses" not in s:
            errs.append("Sentence missing 'clauses'")
            continue
        clause_spans = []
        for c in s["clauses"]:
            st, en = c.get("start"), c.get("end")
            t = c.get("clause_text", "")
            if not isinstance(st, int) or not isinstance(en, int) or st < 0 or en > len(src) or st >= en:
                errs.append(f"Bad clause span: {st},{en}")
                continue
            if src[st:en] != t:
                errs.append(f"Clause text mismatch at {st}:{en}")
            clause_spans.append((st, en))
            ph_spans = []
            for p in c.get("phrases", []):
                pst, pen = p.get("start"), p.get("end")
                ptxt = p.get("text", "")
                if not isinstance(pst, int) or not isinstance(pen, int) or pst < st or pen > en or pst >= pen:
                    errs.append(f"Bad phrase span: {pst},{pen} in clause {st}:{en}")
                    continue
                if src[pst:pen] != ptxt:
                    errs.append(f"Phrase text mismatch at {pst}:{pen}")
                ph_spans.append((pst, pen))
            if not non_overlapping(ph_spans):
                errs.append(f"Overlapping phrase spans in clause {st}:{en}")
        if not non_overlapping(clause_spans):
            errs.append("Overlapping clause spans in sentence")
    return errs


# --------------- LM Studio call ---------------

def lmstudio_chunk(text: str,
                   model: str = LMSTUDIO_MODEL,
                   temperature: float = 0.1,
                   max_tokens: int = 1500) -> Dict[str, Any]:
    prompt = build_prompt(text)
    try:
        resp: ChatCompletion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
    except Exception:
        resp: ChatCompletion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Return valid JSON only. Do not wrap in code fences."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )

    content = resp.choices[0].message.content or ""
    data = extract_json(content)
    errors = validate_result(text, data)
    if errors:
        print("Validator notes:")
        for e in errors:
            print(" -", e)
        # Helpful to see what the model sent
        print("\nRaw model output (trimmed):")
        print(content[:800])
    return data


# --------------- pretty print ---------------

def pretty_print(result: Dict[str, Any]):
    print(f"INPUT: {result.get('text', '')}\n")
    for si, s in enumerate(result.get("sentences", []), 1):
        print(f"Sentence {si}: {s.get('sentence_text', '')}")
        for ci, c in enumerate(s.get("clauses", []), 1):
            print(f"  Clause {ci}: {c.get('clause_text', '')}")
            for p in c.get("phrases", []):
                print(f"    [{p.get('type','')}] {p.get('text','')}")
        print()


# --------------- demo ---------------
if __name__ == "__main__":
    samples = [
        "we ran out of time yesterday afternoon",
        "SO sorry--we ran out of time yesterday afternoon and Dr. Person1 had",
        "The good doctor is swamped in a emails and she wants to make sure he sees it",
        "for the middle of MM/DD/YYYY in Org3 on a MM/DD/YYYY morning.",
        "Can you put an order on for Person2 to have his port flushed for the middle of MM/DD/YYYY in Org3 on a MM/DD/YYYY morning",
        "So no need to schedule anything besides the call with Dr Person2",
        "Currently my lab work is scheduled to be done on MM/DD/YYYY and I see",
        "Currently my lab work is scheduled to be done"
    ]
    for s in samples:
        res = lmstudio_chunk(s)
        pretty_print(res)
