import json
from collections import defaultdict

# === CONFIG ===
input_path = "PV_output_json/codebook_hierarchy.json"
output_file = "PV_output_json/node_names_by_type_with_index.json"

# === LOAD JSON ===
with open(input_path, "r", encoding="utf-8") as f:
    graph_data = json.load(f)

nodes = {node["id"]: node for node in graph_data["nodes"]}
links = graph_data["links"]

# === BUILD CHILDREN MAP ===
parent_to_children = defaultdict(list)
child_to_parent = {}

for link in links:
    parent = link["source"]
    child = link["target"]
    parent_to_children[parent].append(child)
    child_to_parent[child] = parent

# === FIND ROOTS (CODES) ===
roots = [node_id for node_id, node in nodes.items() if node["type"] == "code"]

# === DFS TRAVERSAL TO ASSIGN INDEX ===
indexed_nodes = []
index_by_type = defaultdict(list)

def dfs(node_id, prefix):
    node = nodes[node_id]
    index = prefix
    indexed_nodes.append({
        "id": node_id,
        "type": node["type"],
        "index": index
    })
    index_by_type[node["type"]].append(f"{index}: {node_id}")
    
    children = parent_to_children.get(node_id, [])
    children.sort()  # Optional: keep consistent order
    for i, child_id in enumerate(children, 1):
        dfs(child_id, f"{index}_{i}")

# === RUN DFS FOR EACH ROOT CODE ===
for i, root_id in enumerate(roots, 1):
    dfs(root_id, str(i))

# === SAVE TO FILE ===
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(index_by_type, f, indent=2, ensure_ascii=False)

print(f"✅ Saved underscore-indexed node names to {output_file}")
