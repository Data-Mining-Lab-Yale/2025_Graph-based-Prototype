import json

# ==== FILE PATHS ====
ANNOTATIONS_FILE = "EPPC_output_json/CleanedData/processed_messages_with_annotations.json"
MAPPING_FILE = "EPPC_output_json/Labels/annotation_code_mapping_detailed_corrected.json"
TYPES_FILE = "EPPC_output_json/Labels/split_intents_by_type.json"
OUTPUT_FILE = "EPPC_output_json/CleanedData/processed_messages_with_annotations_with_types.json"

# ==== LOAD FILES ====
with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
    annotations_data = json.load(f)

with open(MAPPING_FILE, "r", encoding="utf-8") as f:
    mapping_data = json.load(f)

with open(TYPES_FILE, "r", encoding="utf-8") as f:
    types_data = json.load(f)

# ==== BUILD LOOKUP FOR TYPES ====
label_type_lookup = {}
for intent_type, items in types_data.items():
    for item in items:
        label_type_lookup[item["label"]] = intent_type

# ==== PROCESS ANNOTATIONS ====
for message in annotations_data:
    for ann in message.get("annotations", []):
        updated_codes = []
        label_types = set()

        for code in ann.get("code", []):
            # Map to codebook label if available
            mapped_label = mapping_data.get(code, {}).get("matched_codebook_label", code)
            updated_codes.append(mapped_label)

            # Assign type if exists
            if mapped_label in label_type_lookup:
                label_types.add(label_type_lookup[mapped_label])

        ann["code"] = updated_codes
        ann["label_type"] = list(label_types)

# ==== SAVE RESULT ====
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(annotations_data, f, ensure_ascii=False, indent=2)

print(f"Processed file saved to {OUTPUT_FILE}")
