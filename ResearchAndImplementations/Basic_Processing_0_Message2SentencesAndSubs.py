# Bethesda_Message2SentencesAndSubs_v3ud.py
# Segment each message into sentences and clause-like subsentences using UD cues with length control.
# Input must follow the processed schema: message, message_id, annotations[text_id,text,code].

import json
from pathlib import Path
import sys

# ---------- paths ----------
INPUT_JSON  = Path("Bethesda_output/Bethesda_processed_messages_with_annotations.json")   # your uploaded file
OUTPUT_JSON = Path("Bethesda_output/Bethesda_messages_with_sentences_and_subsentences.json")

# ---------- settings ----------
MAX_TOKENS_PER_SEG = 18     # hard cap per subsentence
MIN_TOKENS_PER_SEG = 3      # merge short fragments
MIN_CHARS_PER_SEG  = 8

SCONJ_LEX = {
    "because","if","when","while","although","though","since","that",
    "so","but","and","or","as","unless","until","whereas","whether"
}

# ---------- nlp ----------
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_trf")
    except Exception:
        nlp = spacy.load("en_core_web_sm")
except Exception as e:
    print("spaCy is required. pip install spacy && python -m spacy download en_core_web_sm", file=sys.stderr)
    raise

# ---------- helpers ----------
def has_verb_or_modal(span):
    return any(t.pos_ in {"VERB","AUX"} or t.tag_ == "MD" for t in span)

def candidate_cuts(doc):
    cuts = set()
    CLAUSE_DEPS = {"advcl","ccomp","csubj","xcomp","relcl","parataxis"}
    for tok in doc:
        # dependency-based cuts
        if tok.dep_ in CLAUSE_DEPS and tok.i > 0:
            cuts.add(tok.i)
        # coordination starting a new predicate
        if tok.dep_ == "conj" and (tok.pos_ in {"VERB","AUX"} or tok.head.pos_ in {"VERB","AUX"}):
            cuts.add(tok.i)
        # lexical SCONJ and CCONJ
        if tok.text.lower() in SCONJ_LEX and tok.i > 0:
            cuts.add(tok.i)
        if tok.pos_ == "CCONJ" and tok.i > 0:
            cuts.add(tok.i)
        # punctuation where both sides look clause-like
        if tok.text in {",",";",":"} and 1 <= tok.i < len(doc)-1:
            left = doc[max(0, tok.i-8):tok.i]
            right = doc[tok.i+1:min(len(doc), tok.i+9)]
            if has_verb_or_modal(left) and has_verb_or_modal(right):
                cuts.add(tok.i+1)
    return sorted(cuts)

def _split_long(tokens, seg):
    s, e = seg
    length = e - s
    if length <= MAX_TOKENS_PER_SEG:
        return [seg]
    mid = s + length // 2
    # search near mid for a safe cut
    candidates = []
    for i in range(max(s+1, mid-10), min(e-1, mid+11)):
        tok = tokens[i]
        safe = (tok.text in {",",";"} or tok.pos_=="CCONJ" or tok.text.lower() in SCONJ_LEX
                or tok.dep_ in {"conj","advcl","ccomp","xcomp"})
        if safe:
            candidates.append(i)
    if not candidates:
        return [seg]
    cut = min(candidates, key=lambda i: abs(i-mid))
    return _split_long(tokens, (s, cut)) + _split_long(tokens, (cut, e))

def enforce_lengths(tokens, cut_idx_list):
    # initial segments from cut points
    segments = []
    starts = [0] + cut_idx_list
    for s, e in zip(starts, cut_idx_list + [len(tokens)]):
        segments.append((s, e))
    # split long, merge tiny
    refined = []
    for seg in segments:
        refined.extend(_split_long(tokens, seg))
    merged = []
    i = 0
    while i < len(refined):
        s, e = refined[i]
        span = tokens[s:e]
        too_short = (e - s) < MIN_TOKENS_PER_SEG or len(span.text.strip()) < MIN_CHARS_PER_SEG
        if too_short:
            if i+1 < len(refined):
                ns, ne = refined[i+1]
                merged.append((s, ne))
                i += 2
            elif merged:
                ps, pe = merged[-1]
                merged[-1] = (ps, e)
                i += 1
            else:
                merged.append((s, e))
                i += 1
        else:
            merged.append((s, e))
            i += 1
    final = []
    for seg in merged:
        final.extend(_split_long(tokens, seg))
    return final

def sentencize(text):
    doc = nlp(text)
    return [s for s in doc.sents]

def best_ann_match(span_text, annotations):
    from difflib import SequenceMatcher
    best = ""
    best_score = 0.0
    low = span_text.lower()
    for ann in annotations or []:
        cand = ann.get("text","")
        sc = SequenceMatcher(None, low, cand.lower()).ratio()
        if sc > best_score:
            best_score = sc
            best = cand
    return best if best_score >= 0.6 else ""

def process_messages(records):
    out = []
    for msg in records:
        message_text = msg["message"]
        message_id   = msg["message_id"]
        annotations  = msg.get("annotations", [])

        sentences = []
        for i, sent in enumerate(sentencize(message_text)):
            sdoc = nlp(sent.text)
            cuts = candidate_cuts(sdoc)
            segs = enforce_lengths(sdoc, cuts)

            subs = []
            for j, (a, b) in enumerate(segs):
                text_seg = sdoc[a:b].text.strip()
                subs.append({
                    "subsentence_id": f"{message_id}_{i}_{j}",
                    "subsentence": text_seg,
                    "most_close_annotation_span": best_ann_match(text_seg, annotations)
                })

            sentences.append({
                "sentence_id": f"{message_id}_{i}",
                "sentence": sent.text.strip(),
                "most_close_annotation_span": best_ann_match(sent.text, annotations),
                "subsentences": subs
            })

        out.append({
            "message": message_text,
            "message_id": message_id,
            "sentences": sentences
        })
    return out

if __name__ == "__main__":
    data = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    result = process_messages(data)
    OUTPUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Wrote {OUTPUT_JSON}")
