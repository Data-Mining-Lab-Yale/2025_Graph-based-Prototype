import json

FOLDER = "EPPC_output_json/"

# Load input files
with open(FOLDER+"processed_messages_with_annotations.json", "r", encoding="utf-8") as f:
    annotations = json.load(f)

with open(FOLDER+"sentence_index_with_annotation.json", "r", encoding="utf-8") as f:
    sentence_spans = json.load(f)

with open(FOLDER+"subsentence_index_with_annotation.json", "r", encoding="utf-8") as f:
    subsentence_spans = json.load(f)

with open(FOLDER+"annotation_code_mapping_detailed_corrected.json", "r", encoding="utf-8") as f:
    code_metadata = json.load(f)

with open(FOLDER+"node_names_by_type_with_index.json", "r", encoding="utf-8") as f:
    code_ids_by_level = json.load(f)

# Flatten node ID map
# label_to_code_id = {}
# for level, code_dict in code_ids_by_level.items():
#     for code_id, label in code_dict.items():
#         label_to_code_id[label] = {"code_id": code_id, "level": level}
label_to_code_id = {}
for level, entries in code_ids_by_level.items():
    if isinstance(entries, list):
        for entry in entries:
            if ":" in entry:
                code_id, label = entry.split(":", 1)
                label_to_code_id[label.strip()] = {"code_id": code_id.strip(), "level": level}
    elif isinstance(entries, dict):
        for code_id, label in entries.items():
            label_to_code_id[label] = {"code_id": code_id, "level": level}


# Create a mapping from span text to associated codes
span_to_codes = {}
for message in annotations:
    for ann in message.get("annotations", []):
        span_text = ann["text"].strip()
        codes = ann["code"] if isinstance(ann["code"], list) else [ann["code"]]
        span_to_codes[span_text] = codes

# Function to build label info
def build_labels_for_span(span_text):
    results = []
    codes = span_to_codes.get(span_text.strip(), [])
    for code in codes:
        label_entry = {
            "label": code,
            "matched_codebook_label": code_metadata.get(code, {}).get("matched_codebook_label", code),
            "level": code_metadata.get(code, {}).get("level", "unknown"),
        }
        label_info = label_to_code_id.get(label_entry["matched_codebook_label"], {})
        label_entry["code_id"] = label_info.get("code_id", "UNKNOWN")
        results.append(label_entry)
    return results

# Process sentence and subsentence mappings
def process_units(units):
    results = {}
    for unit_id, data in units.items():
        span = data.get("most_close_annotation_span", "").strip()
        if not span:
            continue
        labels = build_labels_for_span(span)
        results[unit_id] = {
            "text": data.get("content", ""),
            "span": span,
            "labels": labels
        }
    return results

sentence_output = process_units(sentence_spans)
subsentence_output = process_units(subsentence_spans)

# Write outputs
with open(FOLDER+"sentence_label_structured.json", "w", encoding="utf-8") as f:
    json.dump(sentence_output, f, ensure_ascii=False, indent=2)

with open(FOLDER+"subsentence_label_structured.json", "w", encoding="utf-8") as f:
    json.dump(subsentence_output, f, ensure_ascii=False, indent=2)

print(f"Files saved: {FOLDER}sentence_label_structured.json, subsentence_label_structured.json")
