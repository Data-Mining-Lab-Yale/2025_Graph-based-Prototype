import json
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

# === CONFIG ===
graph_path = "PV_output_json/codebook_hierarchy.json"

# === LOAD JSON GRAPH DATA ===
with open(graph_path, 'r', encoding='utf-8') as f:
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

# === GET SUBTREE ===
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

# === TREE-LIKE LAYOUT ===
def hierarchy_pos(G, root=None, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
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

# === VISUALIZE ===
# def visualize_subtree(subgraph, title="Hierarchy View"):
#     try:
#         root_node = [n for n in subgraph.nodes() if subgraph.in_degree(n) == 0][0]
#     except IndexError:
#         print("⚠️ Could not find a root node.")
#         return

#     pos = hierarchy_pos(subgraph, root=root_node)
#     node_labels = {n: n for n in subgraph.nodes()}
#     node_colors = []

#     for node in subgraph.nodes(data=True):
#         t = node[1].get("type", "")
#         if t == "code":
#             node_colors.append("skyblue")
#         elif t == "subcode":
#             node_colors.append("lightgreen")
#         else:
#             node_colors.append("lightcoral")

#     plt.figure(figsize=(12, 8))
#     nx.draw(subgraph, pos, labels=node_labels, with_labels=True,
#             node_color=node_colors, node_size=2000, font_size=9,
#             edge_color="gray", font_weight='bold')
#     plt.title(title)
#     plt.axis('off')
#     plt.tight_layout()
#     plt.show()
def visualize_subtree(subgraph, title="Hierarchy View"):
    def hierarchy_pos(G, root=None, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
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

        return _hierarchy_pos(G, root, 0, width, vert_loc, xcenter)

    # === For "all_node" visualization only, add a virtual root ===
    if subgraph == G:
        subgraph = subgraph.copy()  # Work on a copy
        virtual_root = "CODEBOOK_ROOT"
        subgraph.add_node(virtual_root, id=virtual_root, type="virtual")

        # Attach virtual root to all top-level root nodes
        for n in subgraph.nodes():
            if subgraph.in_degree(n) == 0 and n != virtual_root:
                subgraph.add_edge(virtual_root, n)

        root_node = virtual_root
    else:
        # Use the single root in the subgraph
        try:
            root_node = [n for n in subgraph.nodes() if subgraph.in_degree(n) == 0][0]
        except IndexError:
            print("⚠️ Could not find a root node.")
            return

    # === Compute layout ===
    pos = hierarchy_pos(subgraph, root=root_node)

    node_labels = {n: n for n in subgraph.nodes() if n != "CODEBOOK_ROOT"}
    node_colors = []

    for node in subgraph.nodes(data=True):
        t = node[1].get("type", "")
        if t == "code":
            node_colors.append("skyblue")
        elif t == "subcode":
            node_colors.append("lightgreen")
        elif t == "subsubcode":
            node_colors.append("lightcoral")
        elif t == "virtual":
            node_colors.append("white")  # make root invisible or distinguishable
        else:
            node_colors.append("gray")

    plt.figure(figsize=(14, 10))
    nx.draw(subgraph, pos, labels=node_labels, with_labels=True,
            node_color=node_colors, node_size=2000, font_size=9,
            edge_color="gray", font_weight='bold', arrows=True)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# === MAIN LOOP ===
instructions = """
🧭 Instructions:
- Enter a keyword to query a node (e.g., 'Economic', 'Empathy')
- Enter 'all_node' to view the full hierarchy
- Enter 'exit' or press ENTER on a blank line to quit
"""

print(instructions)

while True:
    query = input("🔍 Query node: ").strip()
    
    if query == "" or query.lower() == "exit":
        print("👋 Exiting.")
        break
    elif query.lower() == "all_node":
        visualize_subtree(G, title="Full Hierarchy")
    else:
        subgraph = get_subtree(query)
        if subgraph is not None:
            visualize_subtree(subgraph, title=f"Subtree for '{query}'")
    
    print(instructions)
