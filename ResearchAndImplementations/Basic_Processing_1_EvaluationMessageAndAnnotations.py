import json
from difflib import SequenceMatcher

# === File paths ===
split_file = "Bethesda_output/Bethesda_messages_with_sentences_and_subsentences.json"
annot_file = "Bethesda_output/Bethesda_processed_messages_with_annotations.json"

# === Load files ===
with open(split_file, "r", encoding="utf-8") as f1:
    split_data = json.load(f1)

with open(annot_file, "r", encoding="utf-8") as f2:
    original_data = json.load(f2)

# === Build annotation index by message_id ===
annotation_index = {
    msg["message_id"]: msg.get("annotations", [])
    for msg in original_data
}

# === Statistics counters ===
total_annotations = 0
matched_sentences = 0
matched_subsentences = 0

total_messages = len(split_data)
total_sentences = 0
total_subsentences = 0

# === Match annotations and count segments ===
for message in split_data:
    message_id = message["message_id"]
    sentences = message["sentences"]
    annotations = annotation_index.get(message_id, [])

    total_annotations += len(annotations)
    total_sentences += len(sentences)

    for sentence in sentences:
        total_subsentences += len(sentence["subsentences"])

    for ann in annotations:
        ann_text = ann["text"]
        best_sent_score = 0
        best_subsent_score = 0

        for sentence in sentences:
            # Compare annotation to sentence
            sent_score = SequenceMatcher(None, sentence["sentence"].lower(), ann_text.lower()).ratio()
            best_sent_score = max(best_sent_score, sent_score)

            # Compare annotation to each subsentence
            for subsent in sentence["subsentences"]:
                subsent_score = SequenceMatcher(None, subsent["subsentence"].lower(), ann_text.lower()).ratio()
                best_subsent_score = max(best_subsent_score, subsent_score)

        if best_sent_score >= 0.6:
            matched_sentences += 1
        if best_subsent_score >= 0.6:
            matched_subsentences += 1

# === Final Report ===
print("📊 Evaluation Summary")
print("=" * 40)
print(f"Total Messages:                        {total_messages}")
print(f"Total Sentences:                       {total_sentences}")
print(f"Total Subsentences (Clauses):          {total_subsentences}")
print(f"Total Annotations:                     {total_annotations}")
print()
print(f"Matched Annotations at Sentence Level: {matched_sentences} ({matched_sentences / total_annotations:.2%})")
print(f"Matched Annotations at Subsentence Level: {matched_subsentences} ({matched_subsentences / total_annotations:.2%})")
