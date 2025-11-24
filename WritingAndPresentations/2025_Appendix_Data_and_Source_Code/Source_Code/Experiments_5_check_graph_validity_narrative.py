import os
import json
from pathlib import Path

# === Configuration ===
GRAPH_DIR = "narrative_ego_graphs_with_labels"  # folder with labeled narrative graphs
OUTPUT_LOG = "outputs/narrative_ego_graphs/invalid_graphs_log.json"
FILTERED_DIR = "filtered_narrative_ego_graphs"
os.makedirs(FILTERED_DIR, exist_ok=True)

# === Initialize trackers ===
graph_path = Path(GRAPH_DIR)
log_path = Path(OUTPUT_LOG)
valid_graphs = []
invalid_graphs = []

# === Initialize trackers ===
graph_path = Path(GRAPH_DIR)
log_path = Path(OUTPUT_LOG)
valid_graphs = []
invalid_graphs = []

# === Validate function ===
def is_valid_narrative_graph(graph):
    # Nodes must exist and contain 'id' and 'label'
    nodes = graph.get("nodes", [])
    if not isinstance(nodes, list) or not nodes:
        return False, "Missing or empty 'nodes' list"

    node_ids = set()
    for node in nodes:
        if not isinstance(node, dict) or "id" not in node or "label" not in node:
            return False, "Node missing 'id' or 'label'"
        if not isinstance(node["label"], str) or not node["label"].strip():
            return False, "Invalid or empty label in node"
        node_ids.add(node["id"])

    # Edges must connect valid node ids
    for edge in graph.get("edges", []):
        if "source" not in edge or "target" not in edge:
            return False, "Edge missing 'source' or 'target'"
        if edge["source"] not in node_ids or edge["target"] not in node_ids:
            return False, f"Edge references unknown node: {edge}"

    # Must have a center_id that appears in the node list
    if "center_id" not in graph or graph["center_id"] not in node_ids:
        return False, "Missing or invalid 'center_id'"

    # Must contain a label
    if "label" not in graph or not isinstance(graph["label"], str):
        return False, "Missing or invalid label"

    return True, None

# === Process each graph file ===
for file in graph_path.glob("*.json"):
    try:
        with open(file, "r", encoding="utf-8") as f:
            graph = json.load(f)

        valid, reason = is_valid_narrative_graph(graph)
        if not valid:
            raise ValueError(reason)

    except Exception as e:
        invalid_graphs.append({
            "file": file.name,
            "reason": str(e)
        })
    else:
        valid_graphs.append(file.name)

# === Save log ===
log = {
    "valid_graphs": valid_graphs,
    "invalid_graphs": invalid_graphs,
    "summary": {
        "total": len(valid_graphs) + len(invalid_graphs),
        "valid": len(valid_graphs),
        "invalid": len(invalid_graphs)
    }
}
os.makedirs(log_path.parent, exist_ok=True)
with open(log_path, "w", encoding="utf-8") as f:
    json.dump(log, f, indent=2)

# === Copy valid graphs ===
copied = 0
for fname in valid_graphs:
    src = graph_path / fname
    dst = Path(FILTERED_DIR) / fname
    with open(src, "rb") as sf, open(dst, "wb") as df:
        df.write(sf.read())
    copied += 1

# === Final message ===
print(f"✅ Check complete. Log saved to: {OUTPUT_LOG}")
print(f"🟢 Valid graphs: {len(valid_graphs)}")
print(f"🔴 Invalid graphs: {len(invalid_graphs)}")
if invalid_graphs:
    print(f"⚠️ First few issues: {[e['file'] for e in invalid_graphs[:5]]}")
print(f"📂 Copied {copied} valid graphs to '{FILTERED_DIR}/'")