import os
import json
from pathlib import Path

# === Graph type configuration ===
# TYPE = "predicate"  # Change to: "weighted", "anchored", etc.
# TYPE = "weighted"
TYPE = "anchored"

# === Directory and log paths ===
GRAPH_DIR = f"srl_graphs_{TYPE}_with_labels"
OUTPUT_LOG = f"outputs/srl_graphs_{TYPE}/invalid_graphs_log.json"
FILTERED_DIR = f"filtered_srl_graphs_{TYPE}"
os.makedirs(FILTERED_DIR, exist_ok=True)

# === Initialize trackers ===
graph_path = Path(GRAPH_DIR)
log_path = Path(OUTPUT_LOG)
valid_graphs = []
invalid_graphs = []

# === Process each graph file ===
for file in graph_path.glob("*.json"):
    try:
        with open(file, "r", encoding="utf-8") as f:
            graph = json.load(f)

        nodes = graph.get("nodes", [])
        if not isinstance(nodes, list) or len(nodes) == 0:
            raise ValueError("Missing or empty 'nodes' list")

        tokens = [n["label"] for n in nodes if isinstance(n, dict) and "label" in n]
        if not tokens or not all(isinstance(t, str) and t.strip() for t in tokens):
            raise ValueError("No valid token labels found")

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
with open(log_path, "w", encoding="utf-8") as f:
    json.dump(log, f, indent=2)

# === Copy valid files to filtered folder with cleaned names ===
copied = 0
for fname in valid_graphs:
    src_path = graph_path / fname

    # Remove suffix like _srl, _predicate, etc.
    subsentence_id = Path(fname).stem
    for suffix in ["_srl", "_predicate", "_anchored", "_weighted", "_focus"]:
        if subsentence_id.endswith(suffix):
            subsentence_id = subsentence_id.rsplit("_", 1)[0]
            break

    dst_path = Path(FILTERED_DIR) / f"{subsentence_id}.json"

    if src_path.exists():
        with open(src_path, "rb") as src_file, open(dst_path, "wb") as dst_file:
            dst_file.write(src_file.read())
        copied += 1

# === Final message ===
print(f"✅ Check complete. Log saved to: {OUTPUT_LOG}")
print(f"🟢 Valid graphs: {len(valid_graphs)}")
print(f"🔴 Invalid graphs: {len(invalid_graphs)}")
if invalid_graphs:
    print(f"⚠️ First few issues: {[e['file'] for e in invalid_graphs[:5]]}")
print(f"📂 Copied {copied} valid graphs to '{FILTERED_DIR}/'")
