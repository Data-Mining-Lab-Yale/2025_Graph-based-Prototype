import json
from pathlib import Path
FOLDER = "EPPC_output_json/"

# Load the full message dataset
with open(FOLDER+"messages_with_sentences_and_subsentences.json", "r", encoding="utf-8") as f:
    messages = json.load(f)

# Initialize mappings
message_map = {}
sentence_map = {}
subsentence_map = {}

# Also initialize filtered versions
message_with_anno = {}
sentence_with_anno = {}
subsentence_with_anno = {}

# Process the hierarchical structure
for msg in messages:
    msg_id = str(msg["message_id"])
    message_map[msg_id] = {
        "content": msg["message"],
        "most_close_annotation_span": ""
    }

    has_msg_level_anno = False
    for sent in msg["sentences"]:
        sent_id = sent["sentence_id"]
        sentence_map[sent_id] = {
            "content": sent["sentence"],
            "most_close_annotation_span": sent.get("most_close_annotation_span", "")
        }
        if sent.get("most_close_annotation_span"):
            sentence_with_anno[sent_id] = sentence_map[sent_id]
            has_msg_level_anno = True

        for subsent in sent.get("subsentences", []):
            subsent_id = subsent["subsentence_id"]
            subsentence_map[subsent_id] = {
                "content": subsent["subsentence"],
                "most_close_annotation_span": subsent.get("most_close_annotation_span", "")
            }
            if subsent.get("most_close_annotation_span"):
                subsentence_with_anno[subsent_id] = subsentence_map[subsent_id]
                has_msg_level_anno = True

    if has_msg_level_anno:
        message_with_anno[msg_id] = message_map[msg_id]

# Save all outputs
outputs = {
    FOLDER+"message_index_mapping.json": message_map,
    FOLDER+"sentence_index_mapping.json": sentence_map,
    FOLDER+"subsentence_index_mapping.json": subsentence_map,
    FOLDER+"message_index_with_annotation.json": message_with_anno,
    FOLDER+"sentence_index_with_annotation.json": sentence_with_anno,
    FOLDER+"subsentence_index_with_annotation.json": subsentence_with_anno,
}

for fname, content in outputs.items():
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=2)

print("Files saved.")
