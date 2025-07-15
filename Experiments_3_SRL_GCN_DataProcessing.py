import os
import json
from pathlib import Path

# === Configuration ===
GRAPH_DIR = "outputs/srl_graphs_weighted/subsentence_subcode/json"
LABEL_FILE = "EPPC_output_json/subsentence_subcode_labels.json"
TEXT_SOURCE_FILE = "EPPC_output_json/subsentence_subcode_labels.json"  # Can be the same as label file
OUTPUT_DIR = "srl_graphs_weighted_with_labels"

# === Setup paths ===
graph_path = Path(GRAPH_DIR)
label_path = Path(LABEL_FILE)
text_path = Path(TEXT_SOURCE_FILE)
output_path = Path(OUTPUT_DIR)
output_path.mkdir(parents=True, exist_ok=True)

# === Load label map ===
with open(label_path, "r", encoding="utf-8") as f:
    label_data = json.load(f)

label_map = {}
for subsentence_id, entry in label_data.items():
    for label_info in entry.get("labels", []):
        if label_info.get("level") == "subcode":
            label_map[subsentence_id] = label_info["label"]
            break  # Use first subcode

# === Load text if needed ===
with open(text_path, "r", encoding="utf-8") as f:
    text_data = json.load(f)

print(f"✅ Loaded {len(label_map)} subcode labels.")

# === Process and save updated graphs ===
count = 0
skipped = 0
errors = []

for file in graph_path.glob("*.json"):
    subsentence_id = file.stem
    try:
        with open(file, "r", encoding="utf-8") as f:
            graph = json.load(f)

        if subsentence_id not in label_map:
            skipped += 1
            continue

        # Add label and text
        graph["label"] = label_map[subsentence_id]
        graph["text"] = text_data.get(subsentence_id, {}).get("text", "[text not found]")

        # Save
        with open(output_path / file.name, "w", encoding="utf-8") as f:
            json.dump(graph, f, indent=2)
        count += 1

    except Exception as e:
        errors.append({"file": file.name, "error": str(e)})

# === Summary ===
print(f"✅ Updated and saved {count} graphs to '{OUTPUT_DIR}/'")
print(f"⚠️ Skipped: {skipped} (no label) | ❌ Errors: {len(errors)}")

if errors:
    with open("outputs/srl_graphs_weighted/data_processing_errors.json", "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2)
    print("📝 Error log written to: data_processing_errors.json")
