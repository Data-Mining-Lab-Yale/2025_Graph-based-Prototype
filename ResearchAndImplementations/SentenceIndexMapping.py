import json

TARGET_PROJECT_NAME = "EPPC" #TARGET_PROJECT_NAME +"_output_json/
# TARGET_PROJECT_NAME = "PV"

# Load the input JSON file
with open(TARGET_PROJECT_NAME +"_output_json/messages_with_sentences_and_subsentences.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Initialize mappings
message_level = {}
sentence_level = {}
subsentence_level = {}

# Create mappings
for msg_idx, item in enumerate(data):
    msg_content = item.get("content", "")
    msg_span = item.get("most_close_annotation_span", "")
    message_level[f"{msg_idx}"] = {
        "content": msg_content,
        "most_close_annotation_span": msg_span
    }

    for sent_idx, sentence in enumerate(item.get("sentences", [])):
        sent_content = sentence.get("content", "")
        sent_span = sentence.get("most_close_annotation_span", "")
        sentence_level[f"{msg_idx}_{sent_idx}"] = {
            "content": sent_content,
            "most_close_annotation_span": sent_span
        }

        for sub_idx, subsentence in enumerate(sentence.get("subsentences", [])):
            sub_content = subsentence.get("content", "")
            sub_span = subsentence.get("most_close_annotation_span", "")
            subsentence_level[f"{msg_idx}_{sent_idx}_{sub_idx}"] = {
                "content": sub_content,
                "most_close_annotation_span": sub_span
            }

# Filter non-empty annotation span entries
message_filtered = {k: v for k, v in message_level.items() if v["most_close_annotation_span"].strip()}
sentence_filtered = {k: v for k, v in sentence_level.items() if v["most_close_annotation_span"].strip()}
subsentence_filtered = {k: v for k, v in subsentence_level.items() if v["most_close_annotation_span"].strip()}

# Save to JSON files
with open(TARGET_PROJECT_NAME +"_output_json/message_index_mapping.json", "w", encoding="utf-8") as f:
    json.dump(message_level, f, ensure_ascii=False, indent=2)
with open(TARGET_PROJECT_NAME +"_output_json/message_index_with_annotation.json", "w", encoding="utf-8") as f:
    json.dump(message_filtered, f, ensure_ascii=False, indent=2)

with open(TARGET_PROJECT_NAME +"_output_json/sentence_index_mapping.json", "w", encoding="utf-8") as f:
    json.dump(sentence_level, f, ensure_ascii=False, indent=2)
with open(TARGET_PROJECT_NAME +"_output_json/sentence_index_with_annotation.json", "w", encoding="utf-8") as f:
    json.dump(sentence_filtered, f, ensure_ascii=False, indent=2)

with open(TARGET_PROJECT_NAME +"_output_json/subsentence_index_mapping.json", "w", encoding="utf-8") as f:
    json.dump(subsentence_level, f, ensure_ascii=False, indent=2)
with open(TARGET_PROJECT_NAME +"_output_json/subsentence_index_with_annotation.json", "w", encoding="utf-8") as f:
    json.dump(subsentence_filtered, f, ensure_ascii=False, indent=2)

print("All files saved.")
