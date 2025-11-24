import os
import json
from pathlib import Path

# === Configuration ===
GRAPH_DIR = "outputs/narrative_ego_graphs/subsentence_subcode/json"  # folder with narrative graphs
LABEL_FILE = "EPPC_output_json/subsentence_subcode_labels.json"
OUTPUT_DIR = "narrative_ego_graphs_with_labels"

# === Setup paths ===
graph_path = Path(GRAPH_DIR)
label_path = Path(LABEL_FILE)
output_path = Path(OUTPUT_DIR)
output_path.mkdir(parents=True, exist_ok=True)

# === Load label and text map ===
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

print(f"✅ Loaded {len(label_map)} labels from annotation.")

# === Process narrative graphs ===
count, skipped, errors = 0, 0, []

for file in graph_path.glob("*.json"):
    try:
        with open(file, "r", encoding="utf-8") as f:
            graph = json.load(f)

        center_id = graph.get("center_id")
        if not center_id or center_id not in label_map:
            skipped += 1
            continue

        graph["label"] = label_map[center_id]
        graph["text"] = text_map.get(center_id, "[text not found]")

        with open(output_path / file.name, "w", encoding="utf-8") as f:
            json.dump(graph, f, indent=2)
        count += 1

    except Exception as e:
        errors.append({"file": file.name, "error": str(e)})

# === Summary ===
print(f"✅ Saved {count} narrative graphs with labels to '{OUTPUT_DIR}/'")
print(f"⚠️ Skipped {skipped} graphs (no label) | ❌ {len(errors)} errors")

if errors:
    with open(output_path / "data_processing_errors.json", "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2)
