import json
import os
import re
from collections import defaultdict, deque
import networkx as nx
import matplotlib.pyplot as plt

# === CONFIG ===
graph_path = "PV_output_json/codebook_hierarchy.json"
index_path = "PV_output_json/node_names_by_type_with_index.json"
output_folder = "PV_output_json/subtree_images"

# === PREP OUTPUT FOLDER ===
os.makedirs(output_folder, exist_ok=True)

# === LOAD GRAPH AND INDEX ===
with open(graph_path, 'r', encoding='utf-8') as f:
    graph_data = json.load(f)

with open(index_path, 'r', encoding='utf-8') as f:
    index_by_type = json.load(f)

nodes = graph_data["nodes"]
links = graph_data["links"]
node_map = {n["id"]: n for n in nodes}

# Build graph
G = nx.DiGraph()
node_lookup = {n["id"].lower(): n["id"] for n in nodes}
for n in nodes:
    G.add_node(n["id"], **n)
for l in links:
    G.add_edge(l["source"], l["target"])

# === Build ID → Index Map from Parsed Index File ===
id_to_index = {}
for type_list in index_by_type.values():
    for item in type_list:
        index, node_id = item.split(": ", 1)
        id_to_index[node_id] = index

# === Helper to sanitize file names ===
def sanitize_filename(name):
    return re.sub(r'[<>:"/\\|?*]', '_', name)

# === SUBTREE EXTRACTION ===
def get_subtree(node_id):
    if node_id not in G:
        return None
    subtree_nodes = set()
    queue = deque([node_id])
    while queue:
        current = queue.popleft()
        subtree_nodes.add(current)
        for child in G.successors(current):
            if child not in subtree_nodes:
                queue.append(child)
    return G.subgraph(subtree_nodes)

# === VISUALIZATION ===
def visualize_and_save(subgraph, filename):
    def hierarchy_pos(G, root=None, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
        """
        If the graph is a tree this will return the positions to plot it in a hierarchical layout.
        """
        def _hierarchy_pos(G, root, left, right, vert_loc, x_pos, pos=None, parent=None):
            if pos is None:
                pos = {root: (x_pos, vert_loc)}
            else:
                pos[root] = (x_pos, vert_loc)
            children = list(G.successors(root))
            if len(children) != 0:
                dx = (right - left) / len(children)
                next_x = left + dx / 2
                for child in children:
                    pos = _hierarchy_pos(G, child, left=next_x - dx / 2, right=next_x + dx / 2,
                                         vert_loc=vert_loc - vert_gap, x_pos=next_x, pos=pos, parent=root)
                    next_x += dx
            return pos

        if root is None:
            root = [n for n in G.nodes() if G.in_degree(n) == 0]
            if len(root) == 0:
                raise ValueError("No root found for layout")
            root = root[0]
        return _hierarchy_pos(G, root, 0, width, vert_loc, xcenter)

    plt.figure(figsize=(12, 8))
    
    # Tree layout
    try:
        root_node = [n for n in subgraph.nodes() if subgraph.in_degree(n) == 0][0]
    except IndexError:
        print(f"Skipping {filename} (no root found)")
        return

    pos = hierarchy_pos(subgraph, root=root_node)
    
    node_labels = {n: n for n in subgraph.nodes()}
    node_colors = []

    for node in subgraph.nodes(data=True):
        t = node[1].get("type", "")
        if t == "code":
            node_colors.append("skyblue")
        elif t == "subcode":
            node_colors.append("lightgreen")
        else:
            node_colors.append("lightcoral")

    nx.draw(subgraph, pos, labels=node_labels, with_labels=True,
            node_color=node_colors, node_size=2000, font_size=9,
            edge_color="gray", font_weight='bold', arrows=True)
    plt.title(filename.replace("_", " "))
    plt.axis('off')
    plt.tight_layout()
    
    safe_path = os.path.join(output_folder, sanitize_filename(f"{filename}.png"))
    plt.savefig(safe_path)
    plt.close()

# === MAIN LOOP ===
for node_id, index in id_to_index.items():
    subgraph = get_subtree(node_id)
    if subgraph is None or len(subgraph) == 0:
        continue
    filename = f"{index}_{node_id}"
    visualize_and_save(subgraph, filename)

print(f"✅ Saved subtree images to: {output_folder}")
