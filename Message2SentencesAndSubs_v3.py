# Message2SentencesAndSubs_UD_len.py
# Segments each message into sentences and clause-like subsentences using UD cues with length control.

import json
from pathlib import Path
import re
import sys

# -------------- paths --------------
INPUT_JSON  = r"EPPC_output_json/CleanedData/processed_annotations_with_types.json"   # change to your path
OUTPUT_JSON = r"EPPC_output_json/CleanedData/messages_with_sentences_and_subsentences_v3ud.json"

# -------------- settings --------------
MAX_TOKENS_PER_SEG = 18     # hard cap for a subsentence
MIN_TOKENS_PER_SEG = 3      # merge short fragments
MIN_CHARS_PER_SEG  = 8

SCONJ_LEX = {
    "because","if","when","while","although","though","since","that",
    "so","but","and","or","as","unless","until","whereas","whether"
}

# -------------- nlp --------------
try:
    import spacy
    # prefer a better parser if available
    try:
        nlp = spacy.load("en_core_web_trf")
    except Exception:
        nlp = spacy.load("en_core_web_sm")
except Exception as e:
    print("spaCy is required. pip install spacy && python -m spacy download en_core_web_sm", file=sys.stderr)
    raise

# -------------- helpers --------------
def has_verb_or_modal(span):
    return any(t.pos_ in {"VERB","AUX"} or t.tag_ in {"MD"} for t in span)

def candidate_cuts(doc):
    cuts = set()
    # dependency-based cuts
    CLAUSE_DEPS = {"advcl","ccomp","csubj","xcomp","relcl","parataxis"}
    for i,tok in enumerate(doc):
        if tok.dep_ in CLAUSE_DEPS and tok.i > 0:
            cuts.add(tok.i)
        # coordination that starts a new verbal predicate
        if tok.dep_ == "conj" and (tok.pos_ in {"VERB","AUX"} or tok.head.pos_ in {"VERB","AUX"}):
            cuts.add(tok.i)
        # lexical SCONJ and coordinating CCONJ
        if tok.text.lower() in SCONJ_LEX and tok.i > 0:
            cuts.add(tok.i)
        if tok.pos_ in {"CCONJ"} and tok.i > 0:
            cuts.add(tok.i)

        # punctuation candidates (comma, semicolon, colon) if both sides look clause-like
        if tok.text in {",",";",":"} and 1 <= tok.i < len(doc)-1:
            left = doc[max(0, tok.i-8):tok.i]
            right = doc[tok.i+1:min(len(doc), tok.i+9)]
            if has_verb_or_modal(left) and has_verb_or_modal(right):
                cuts.add(tok.i+1)
    return sorted(cuts)

def enforce_lengths(tokens, cut_idx_list):
    # turn cut indexes into segments, then refine by max and min lengths
    segments = []
    starts = [0] + cut_idx_list
    for s, e in zip(starts, cut_idx_list + [len(tokens)]):
        segments.append([s, e])

    # split long segments at safe places near midpoints
    def split_long(seg):
        s,e = seg
        length = e - s
        if length <= MAX_TOKENS_PER_SEG:
            return [seg]
        mid = s + length // 2
        # search outward for a safe cut close to mid
        candidates = []
        for i in range(max(s+1, mid-10), min(e-1, mid+11)):
            tok = tokens[i]
            safe = (tok.text in {",",";"} or tok.pos_=="CCONJ" or tok.text.lower() in SCONJ_LEX
                    or tok.dep_ in {"conj","advcl","ccomp","xcomp"})
            if safe:
                candidates.append(i)
        if not candidates:
            return [seg]  # give up
        # choose the candidate closest to mid
        cut = min(candidates, key=lambda i: abs(i-mid))
        left, right = [s, cut], [cut, e]
        return split_long(left) + split_long(right)

    refined = []
    for seg in segments:
        refined.extend(split_long(seg))

    # merge very short fragments
    merged = []
    i = 0
    while i < len(refined):
        s,e = refined[i]
        span = tokens[s:e]
        if (e - s) < MIN_TOKENS_PER_SEG or len(span.text.strip()) < MIN_CHARS_PER_SEG:
            if i+1 < len(refined):
                # merge with right neighbor
                ns, ne = refined[i+1]
                merged.append([s, ne])
                i += 2
            elif merged:
                # merge with previous if no right neighbor
                ps, pe = merged[-1]
                merged[-1] = [ps, e]
                i += 1
            else:
                merged.append([s, e])
                i += 1
        else:
            merged.append([s, e])
            i += 1

    # final pass to ensure max length after merges
    final = []
    for seg in merged:
        final.extend(split_long(seg))
    return final

def sentencize(text):
    # use spaCy sentence boundaries
    doc = nlp(text)
    return [sent for sent in doc.sents]

def best_ann_match(span_text, annotations):
    # light fuzzy match for display, same as your current idea
    from difflib import SequenceMatcher
    best = ""
    best_score = 0.0
    low = span_text.lower()
    for ann in annotations or []:
        sc = SequenceMatcher(None, low, ann.get("text","").lower()).ratio()
        if sc > best_score:
            best_score = sc
            best = ann.get("text","")
    return best if best_score >= 0.6 else ""

# -------------- main --------------
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
            for j,(a,b) in enumerate(segs):
                subs.append({
                    "subsentence_id": f"{message_id}_{i}_{j}",
                    "subsentence": sdoc[a:b].text.strip(),
                    "most_close_annotation_span": best_ann_match(sdoc[a:b].text, annotations)
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
    data = json.loads(Path(INPUT_JSON).read_text(encoding="utf-8"))
    result = process_messages(data)
    Path(OUTPUT_JSON).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Wrote {OUTPUT_JSON}")
