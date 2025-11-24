import os
import json

# === Configuration ===
MAPPING_FILE = "EPPC_output_json/annotation_code_mapping_detailed_corrected.json"
ERROR_FILES = [
    "results/results_dep_gcn/dep_gcn_errors.json",
    "results/results_sentence/sentence_errors.json",
    "results/results_srl_gcn_weighted/srl_gcn_weighted_errors.json",
    "results/results_subsentence/subsentence_errors.json"
]

# === Load mapping file ===
with open(MAPPING_FILE, "r", encoding="utf-8") as f:
    mapping_data = json.load(f)

def map_label(label):
    """Return the mapped codebook label, or placeholder if unknown."""
    return mapping_data.get(label, {}).get("matched_codebook_label", f"[UNKNOWN:{label}]")

# === Convert each error file ===
for file_path in ERROR_FILES:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    errors = data.get("errors", [])
    standardized = []
    for entry in errors:
        true_label = entry.get("true_label")
        pred_label = entry.get("pred_label")

        standardized.append({
            "id": entry.get("id", None),
            "text": entry.get("text"),
            "true_label": true_label,
            "true_codebook": map_label(true_label),
            "pred_label": pred_label,
            "pred_codebook": map_label(pred_label),
            "correct": entry.get("correct", None)
        })

    output_path = file_path.replace(".json", "_standardized.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(standardized, f, indent=2)

    print(f"✅ Standardized file saved: {output_path}")
