import json
from collections import defaultdict

# === CONFIG ===
input_path = "PV_output_json/codebook_hierarchy.json"
output_file = "PV_output_json/node_names_by_type.json"

# === LOAD JSON ===
with open(input_path, "r", encoding="utf-8") as f:
    graph_data = json.load(f)

# === GROUP NODES BY TYPE ===
node_groups = defaultdict(list)
for node in graph_data["nodes"]:
    node_type = node.get("type", "unknown")
    node_groups[node_type].append(node["id"])

# === SAVE TO FILE ===
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(node_groups, f, indent=2, ensure_ascii=False)

print(f"✅ Saved node names to {output_file}")
