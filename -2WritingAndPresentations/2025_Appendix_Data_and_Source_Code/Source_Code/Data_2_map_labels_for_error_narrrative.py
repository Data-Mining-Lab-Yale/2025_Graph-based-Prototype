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

# === FIND ALL *_errors.json FILES ===
error_files = glob.glob("results/results_*/*_errors.json")

for file_path in error_files:
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            entries = json.load(f)
        except Exception as e:
            print(f"❌ Failed to load {file_path}: {e}")
            continue

    if not isinstance(entries, list):
        print(f"⚠️ Skipping non-list content in: {file_path}")
        print(f"🔎 Content preview: {str(entries)[:200]}")
        continue

    standardized = []
    for entry in entries:
        if not isinstance(entry, dict):
            print(f"⚠️ Skipping non-dict entry in {file_path}: {entry}")
            continue

        # Handle possible mismatches in field names
        true_label = entry.get("true_label") or entry.get("gold_label") or entry.get("label")
        pred_label = entry.get("pred_label") or entry.get("predicted_label") or entry.get("prediction")

        text = entry.get("text") or entry.get("original_text")
        uid = entry.get("id") or entry.get("graph_id") or entry.get("center_id")

        standardized.append({
            "id": uid,
            "text": text,
            "true_label": true_label,
            "true_codebook": map_label(true_label),
            "pred_label": pred_label,
            "pred_codebook": map_label(pred_label),
            "correct": False
        })

    output_path = file_path.replace(".json", "_standardized.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(standardized, f, indent=2)

    print(f"✅ Saved: {output_path}")
