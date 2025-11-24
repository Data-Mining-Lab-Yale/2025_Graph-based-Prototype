import os
import json
import glob

# === CONFIGURATION ===
MAPPING_FILE = "EPPC_output_json/annotation_code_mapping_detailed_corrected.json"

# === LOAD MAPPING FILE ===
with open(MAPPING_FILE, "r", encoding="utf-8") as f:
    mapping_data = json.load(f)

def map_label(label):
    """Return codebook label for an annotation label, or placeholder if unknown."""
    return mapping_data.get(label, {}).get("matched_codebook_label", f"[UNKNOWN:{label}]")

# === FIND ALL *_errors.json FILES (top-level only) ===
error_files = glob.glob("results/results_*/*_errors.json")

# === PROCESS EACH ERROR FILE ===
for file_path in error_files:
    is_error_file = "_errors.json" in os.path.basename(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            errors = json.load(f)
            if not isinstance(errors, list) or not isinstance(errors[0], dict):
                print(f"⚠️ Skipping {file_path}: unexpected format.")
                continue
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            continue

    standardized = []
    for entry in errors:
        true_label = entry.get("true_label")
        pred_label = entry.get("predicted_label") or entry.get("pred_label")

        # Safely determine correctness
        if is_error_file:
            correct = False
        else:
            correct = entry.get("correct", true_label == pred_label)

        standardized.append({
            "id": entry.get("id", None),
            "text": entry.get("text"),
            "true_label": true_label,
            "true_codebook": map_label(true_label),
            "pred_label": pred_label,
            "pred_codebook": map_label(pred_label),
            "correct": correct
        })

    # Save next to original file
    output_path = file_path.replace(".json", "_standardized.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(standardized, f, indent=2)

    print(f"✅ Standardized file saved: {output_path}")
