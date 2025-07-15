import os
import json
from pathlib import Path

# === Configuration ===
GRAPH_DIR = "srl_graphs_with_labels"  # Folder with your <subsentence>.json graph files
OUTPUT_LOG = "invalid_graphs_log.json"

# === Initialize ===
graph_path = Path(GRAPH_DIR)
log_path = Path(OUTPUT_LOG)
valid_graphs = []
invalid_graphs = []

# === Scan and validate each graph file ===
# === Scan and validate each graph file ===
for file in graph_path.glob("*.json"):
    try:
        with open(file, "r", encoding="utf-8") as f:
            graph = json.load(f)

        raw_nodes = graph.get("nodes", [])
        if not isinstance(raw_nodes, list) or len(raw_nodes) == 0:
            raise ValueError("Missing or empty 'nodes' list")

        tokens = [n["label"] for n in raw_nodes if isinstance(n, dict) and "label" in n]
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

# === Done ===
print(f"✅ Check complete. Log saved to: {OUTPUT_LOG}")
print(f"🟢 Valid graphs: {len(valid_graphs)}")
print(f"🔴 Invalid graphs: {len(invalid_graphs)}")
if invalid_graphs:
    print(f"⚠️ First few issues: {[e['file'] for e in invalid_graphs[:5]]}")


# === Optional: Copy valid graphs to a filtered folder ===
FILTERED_DIR = "filtered_srl_graphs"
os.makedirs(FILTERED_DIR, exist_ok=True)

copied = 0
for fname in valid_graphs:
    src_path = graph_path / fname
    dst_path = Path(FILTERED_DIR) / fname
    if src_path.exists():
        with open(src_path, "rb") as src_file, open(dst_path, "wb") as dst_file:
            dst_file.write(src_file.read())
        copied += 1

print(f"📂 Copied {copied} valid graphs to '{FILTERED_DIR}/'")