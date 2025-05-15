import json
import os
from collections import defaultdict, deque
import networkx as nx
import matplotlib.pyplot as plt

# === CONFIG ===
input_path = "EPPC_output_json/codebook_hierarchy.json"

# === LOAD JSON GRAPH DATA ===
with open(input_path, 'r', encoding='utf-8') as f:
    graph_data = json.load(f)

nodes = graph_data["nodes"]
links = graph_data["links"]

# === BUILD GRAPH ===
G = nx.DiGraph()
node_lookup = {}

for node in nodes:
    node_lookup[node["id"].lower()] = node
    G.add_node(node["id"], **node)

for link in links:
    G.add_edge(link["source"], link["target"])

# === FUNCTION TO GET SUBTREE OF ANY NODE ===
def get_subtree(root_name):
    root_id = None
    search = root_name.strip().lower()
    
    for node_id in node_lookup:
        if search in node_id:
            root_id = node_lookup[node_id]["id"]
            break
    
    if root_id is None:
        print(f"❌ Node containing '{root_name}' not found.")
        return None

    subtree_nodes = set()
    queue = deque([root_id])
    while queue:
        current = queue.popleft()
        subtree_nodes.add(current)
        for child in G.successors(current):
            if child not in subtree_nodes:
                queue.append(child)

    return G.subgraph(subtree_nodes)

# === FUNCTION TO VISUALIZE A SUBTREE ===
def visualize_subtree(subgraph, title="Hierarchy View"):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(subgraph, seed=42)
    node_labels = {node: node for node in subgraph.nodes()}
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
            edge_color="gray", font_weight='bold')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# === INTERFACE FUNCTION ===
def query_and_visualize(keyword):
    if keyword.strip().lower() == "all_node":
        subgraph = G
        title = "Full Hierarchy"
    else:
        subgraph = get_subtree(keyword)
        if subgraph is None:
            return
        title = f"Subtree for '{keyword}'"
    
    visualize_subtree(subgraph, title=title)

# === EXAMPLE USAGE ===
# query_and_visualize("Economic_stability")
# query_and_visualize("all_node")
