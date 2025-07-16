import os
import json
from pathlib import Path

# === Configuration ===
GRAPH_DIR = "outputs/amr_graphs/subsentence_subcode/json"  # folder where AMR JSONs are stored
LABEL_FILE = "EPPC_output_json/subsentence_subcode_labels.json"
TEXT_SOURCE_FILE = "EPPC_output_json/subsentence_subcode_labels.json"  # Can be the same as label file
OUTPUT_DIR = "amr_graphs_with_labels"

# === Setup paths ===
graph_path = Path(GRAPH_DIR)
label_path = Path(LABEL_FILE)
output_path = Path(OUTPUT_DIR)
output_path.mkdir(parents=True, exist_ok=True)

# === Load labels and texts ===
with open(label_path, "r", encoding="utf-8") as f:
    label_data = json.load(f)

label_map = {}
text_map = {}
for subsentence_id, entry in label_data.items():
    text_map[subsentence_id] = entry.get("text", "")
    for label_info in entry.get("labels", []):
        if label_info.get("level") == "subcode":
            label_map[subsentence_id] = label_info["label"]
            break

print(f"✅ Loaded {len(label_map)} labels.")

# === Process and save updated AMR graphs ===
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

        graph["label"] = label_map[subsentence_id]
        graph["text"] = text_map.get(subsentence_id, graph.get("text", "[text not found]"))

        with open(output_path / file.name, "w", encoding="utf-8") as f:
            json.dump(graph, f, indent=2)
        count += 1

    except Exception as e:
        errors.append({"file": file.name, "error": str(e)})

# === Summary ===
print(f"✅ Updated and saved {count} graphs to '{OUTPUT_DIR}/'")
print(f"⚠️ Skipped {skipped} graphs (no label) | ❌ {len(errors)} errors")

if errors:
    with open(output_path / "amr_graphs_with_labels/data_processing_errors.json", "w", encoding="utf-8") as f:_
