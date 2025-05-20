import json
import nltk
from difflib import SequenceMatcher
# import nltk
#   >>> nltk.download('punkt_tab')


# Download sentence tokenizer if not already available
nltk.download('punkt')
nltk.download('punkt_tab')

# === Input/Output Paths ===
input_path = "EPPC_output_json/processed_messages_with_annotations.json"
output_path = "EPPC_output_json/messages_with_sentences_and_annotations.json"

# Load messages
with open(input_path, "r", encoding="utf-8") as f:
    messages = json.load(f)

# Helper: Find closest annotation span using string similarity
def find_closest_annotation(sentence, annotations, threshold=0.6):
    best_match = ""
    best_score = 0
    for ann in annotations:
        score = SequenceMatcher(None, sentence.lower(), ann["text"].lower()).ratio()
        if score > best_score:
            best_score = score
            best_match = ann["text"]
    return best_match if best_score >= threshold else ""

# Segment sentences
segmented = []

for msg in messages:
    message_id = msg["message_id"]
    message_text = msg["message"]
    annotations = msg.get("annotations", [])

    sentence_list = nltk.sent_tokenize(message_text)
    sent_objs = []

    for i, sent in enumerate(sentence_list):
        matched_ann = find_closest_annotation(sent, annotations)
        sent_objs.append({
            "sentence_id": f"{message_id}_{i}",
            "sentence": sent,
            "most_close_annotation_span": matched_ann,
            "subsentences": []  # placeholder for future
        })

    segmented.append({
        "message": message_text,
        "message_id": message_id,
        "sentences": sent_objs
    })

# Save to file
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(segmented, f, indent=2, ensure_ascii=False)

print(f"✅ Output saved to: {output_path}")
