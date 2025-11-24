import json
import nltk
import spacy
from difflib import SequenceMatcher

# === Setup ===
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")

input_path = "EPPC_output_json/processed_messages_with_annotations.json"
output_path = "EPPC_output_json/messages_with_sentences_and_subsentences.json"

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
def extract_clauses(text):
    doc = nlp(text)
    clauses = []
    current = []

    for token in doc:
        current.append(token)
        if token.dep_ in {"ccomp", "advcl", "relcl", "conj", "parataxis"} or token.text in {",", ";"}:
            if current:
                clause = spacy.tokens.Span(doc, current[0].i, token.i + 1)
                clauses.append(clause.text.strip())
                current = []
    if current:
        clause = spacy.tokens.Span(doc, current[0].i, current[-1].i + 1)
        clauses.append(clause.text.strip())

    return [c for c in clauses if len(c) > 1]

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
