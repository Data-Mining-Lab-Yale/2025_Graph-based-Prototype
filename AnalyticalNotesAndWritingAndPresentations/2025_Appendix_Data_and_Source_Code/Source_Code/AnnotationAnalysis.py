import json
from difflib import SequenceMatcher
from collections import Counter
import pandas as pd

TARGET_PROJECT_NAME = "EPPC" #TARGET_PROJECT_NAME +"_output_json/
# TARGET_PROJECT_NAME = "PV"

# === Load data ===
with open(TARGET_PROJECT_NAME +"_output_json/codebook_hierarchy.json", "r", encoding="utf-8") as f_codebook:
    codebook_data = json.load(f_codebook)

with open(TARGET_PROJECT_NAME +"_output_json/processed_messages_with_annotations.json", "r", encoding="utf-8") as f_ann:
    annotations_data = json.load(f_ann)

# === Build lookup: code -> type ("Code" or "Sub-code")
code_type_lookup = {}
code_ids = []

for node in codebook_data.get("nodes", []):
    code_id = node.get("id")
    code_type = node.get("type")
    if code_id:
        code_ids.append(code_id)
        if code_type:
            code_type_lookup[code_id] = code_type

# === Fuzzy match function
def find_best_code_match(label, threshold=0.6):
    best_match = ""
    best_score = 0
    for code in code_ids:
        score = SequenceMatcher(None, label.lower(), code.lower()).ratio()
        if score > best_score:
            best_match = code
            best_score = score
    return best_match if best_score >= threshold else ""

# === Annotated label mapping and frequency counting
detailed_mapping = {}
annotation_label_counter = Counter()
annotation_level_tracker = {}

for message in annotations_data:
    for ann in message.get("annotations", []):
        for code_label in ann.get("code", []):
            matched_code = find_best_code_match(code_label)
            if matched_code:
                level = code_type_lookup.get(matched_code, "Unknown")
                annotation_label_counter[matched_code] += 1
                annotation_level_tracker[matched_code] = level
            else:
                level = "Unknown"
            detailed_mapping[code_label] = {
                "matched_codebook_label": matched_code,
                "level": level
            }

# === Save JSON mapping
with open(TARGET_PROJECT_NAME +"_output_json/annotation_code_mapping_detailed_corrected.json", "w", encoding="utf-8") as f_out:
    json.dump(detailed_mapping, f_out, indent=2, ensure_ascii=False)

# === Build frequency summary DataFrame
df_counts = pd.DataFrame([
    {"Matched Codebook Label": label, "Frequency": freq, "Level": annotation_level_tracker.get(label, "Unknown")}
    for label, freq in annotation_label_counter.items()
])
df_counts = df_counts.sort_values(by="Frequency", ascending=False)

# === Save CSV frequency table
df_counts.to_csv(TARGET_PROJECT_NAME +"_output_json/annotation_code_frequency_summary_corrected.csv", index=False)

print("✅ Saved:")
print("- "+TARGET_PROJECT_NAME +"_output_json/annotation_code_mapping_detailed_corrected.json")
print("- "+TARGET_PROJECT_NAME +"_output_json/annotation_code_frequency_summary_corrected.csv")
