import json
import nltk
import spacy
from difflib import SequenceMatcher

# === Setup ===
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")

input_path = "EPPC_output_json/CleanedData/processed_messages_with_annotations.json"
output_path = "EPPC_output_json/CleanedData/messages_with_sentences_and_subsentences_v2.json"

with open(input_path, "r", encoding="utf-8") as f:
    messages_raw = json.load(f)

# === Similarity Matcher ===
def find_closest_annotation(text, annotations, threshold=0.6):
    best_match = ""
    best_score = 0
    for ann in annotations:
        score = SequenceMatcher(None, text.lower(), ann["text"].lower()).ratio()
        if score > best_score:
            best_score = score
            best_match = ann["text"]
    return best_match if best_score >= threshold else ""

# === Clause/Clause-like Chunking ===
def extract_clauses(text, nlp=None, min_tokens=3):
    """
    Safer clause segmentation:
      - Use clause-head subtrees for ccomp/xcomp/advcl/relcl/parataxis
      - Do NOT split at conj alone
      - Split at strong punctuation (.!?;:)
      - Merge tiny trailing fragments into the previous chunk
      - Trim trailing commas/conjunctions from the right edge
    """
    if nlp is None:
        import spacy
        nlp = spacy.load("en_core_web_sm")

    doc = nlp(text)
    strong_punct = {".", "!", "?", ";", ":"}
    clause_deps = {"ccomp", "xcomp", "advcl", "relcl", "parataxis"}

    # collect candidate spans from clause-head subtrees
    spans = []
    used = [False] * len(doc)

    for tok in doc:
        if tok.dep_ in clause_deps:
            # subtree span covering the whole subordinate clause
            left = min([t.i for t in tok.subtree])
            right = max([t.i for t in tok.subtree])
            spans.append((left, right))

    # add sentence-root span as a fallback backbone
    if doc[:].root is not None:
        root = doc[:].root
        left = min([t.i for t in root.subtree])
        right = max([t.i for t in root.subtree])
        spans.append((left, right))

    # add punctuation-based cuts to ensure we don't merge across sentences
    cut_points = [i for i, t in enumerate(doc) if t.text in strong_punct]
    # convert cut points into spans if we had none
    if not spans:
        start = 0
        for cp in cut_points:
            spans.append((start, cp))
            start = cp + 1
        if start < len(doc):
            spans.append((start, len(doc) - 1))

    # normalize, sort, and merge overlaps
    spans = [(max(0, l), min(len(doc) - 1, r)) for l, r in spans]
    spans = sorted(spans)
    merged = []
    for l, r in spans:
        if not merged:
            merged.append([l, r])
        else:
            if l <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], r)
            else:
                merged.append([l, r])

    # cut merged spans further at strong punctuation boundaries inside
    refined = []
    for l, r in merged:
        start = l
        i = l
        while i <= r:
            if doc[i].text in strong_punct and i >= start:
                refined.append([start, i])
                start = i + 1
            i += 1
        if start <= r:
            refined.append([start, r])

    # tidy edges and enforce a minimum token length
    def clean_right(idx):
        # trim trailing commas or coordinating conjunctions
        while idx >= 0 and (doc[idx].text == "," or doc[idx].dep_ == "cc"):
            idx -= 1
        return idx

    chunks = []
    for l, r in refined:
        r = clean_right(r)
        if r < l:
            continue
        span = doc[l:r+1].text.strip()
        if len(span.split()) >= min_tokens:
            chunks.append((l, r, span))
        else:
            # too short: try to merge into previous if exists
            if chunks:
                pl, pr, ps = chunks[-1]
                pr = r
                span2 = doc[pl:pr+1].text.strip()
                chunks[-1] = (pl, pr, span2)
            else:
                # or keep it if it's the only one
                chunks.append((l, r, span))

    # fallback: if everything got filtered, return the whole sentence
    if not chunks:
        return [text.strip()]
    return [s for (_, _, s) in chunks]


# === Main Processing ===
output_data = []

for msg in messages_raw:
    message_id = msg["message_id"]
    message_text = msg["message"]
    annotations = msg.get("annotations", [])

    sentences_raw = nltk.sent_tokenize(message_text)
    sentences = []

    for i, sent in enumerate(sentences_raw):
        sentence_id = f"{message_id}_{i}"
        sentence_best_match = find_closest_annotation(sent, annotations)

        # Subsentence (clause) extraction
        clauses = extract_clauses(sent)
        if not clauses:
            clauses = [sent]  # fallback to full sentence

        subsentences = []
        for j, clause in enumerate(clauses):
            subsentences.append({
                "subsentence_id": f"{message_id}_{i}_{j}",
                "subsentence": clause,
                "most_close_annotation_span": find_closest_annotation(clause, annotations)
            })

        sentences.append({
            "sentence_id": sentence_id,
            "sentence": sent,
            "most_close_annotation_span": sentence_best_match,
            "subsentences": subsentences
        })

    output_data.append({
        "message": message_text,
        "message_id": message_id,
        "sentences": sentences
    })

# === Save Output ===
with open(output_path, "w", encoding="utf-8") as f_out:
    json.dump(output_data, f_out, indent=2, ensure_ascii=False)

print(f"✅ Output written to: {output_path}")
