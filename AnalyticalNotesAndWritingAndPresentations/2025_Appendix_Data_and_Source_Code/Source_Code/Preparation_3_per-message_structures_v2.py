import json
import networkx as nx
import matplotlib.pyplot as plt
import os

# === Settings ===
INPUT_FILE = "EPPC_output_json/messages_with_sentences_and_subsentences.json"  # Path to your input file
OUTPUT_IMG_DIR = "EPPC_output_json/visualized_graphs_updated"
OUTPUT_JSON_DIR = "EPPC_output_json/structured_graph_json"
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

# === Load Data ===
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# === Helper: Hierarchical Layout ===
def hierarchical_layout(graph, root=None, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
    def _hierarchy_pos(G, root, width=1.0, vert_gap=0.2, vert_loc=0,
                       xcenter=0.5, pos=None, parent=None):
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        children = list(G.successors(root))
        if not children:
            return pos
        dx = width / len(children)
        nextx = xcenter - width / 2 - dx / 2
        for child in children:
            nextx += dx
            pos[child] = (nextx, vert_loc - vert_gap)
            pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                 vert_loc=vert_loc - vert_gap, xcenter=nextx,
                                 pos=pos, parent=root)
        return pos
    return _hierarchy_pos(graph, root, width, vert_gap, vert_loc, xcenter)

# === Process Each Message ===
for i, message in enumerate(data):
    G = nx.DiGraph()
    msg_id = message["message_id"]
    G.add_node(msg_id, label="message", text=message["message"])

    for sent in message["sentences"]:
        sent_id = sent["sentence_id"]
        G.add_node(sent_id, label="sentence", text=sent["sentence"])
        G.add_edge(msg_id, sent_id)

        for sub in sent["subsentences"]:
            sub_id = sub["subsentence_id"]
            G.add_node(sub_id, label="subsentence", text=sub["subsentence"])
            G.add_edge(sent_id, sub_id)

    # Layout
    pos = hierarchical_layout(G, root=msg_id)

    # Color by type
    color_map = {
        "message": "#4C78A8",       # blue
        "sentence": "#72B7B2",      # teal
        "subsentence": "#F58518",   # orange
    }
    node_colors = [color_map[G.nodes[n]["label"]] for n in G.nodes]

    # Labels
    labels = {n: f'{d["label"]}: {d["text"][:40]}...' for n, d in G.nodes(data=True)}

    # Draw and Save PNG
    plt.figure(figsize=(14, 10))
    nx.draw(G, pos, with_labels=True, labels=labels,
            node_size=3000, node_color=node_colors,
            font_size=10, edge_color="gray", arrows=True)
    plt.title(f"Message Structure: {msg_id}", fontsize=14)
    plt.tight_layout()
    img_path = os.path.join(OUTPUT_IMG_DIR, f"message_{i+1:03d}_{msg_id}.png")
    plt.savefig(img_path)
    plt.close()

    # Save JSON
    json_path = os.path.join(OUTPUT_JSON_DIR, f"message_{i+1:03d}_{msg_id}.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(nx.node_link_data(G), jf, ensure_ascii=False, indent=2)

print(f"✅ Finished! Saved {len(data)} PNGs to '{OUTPUT_IMG_DIR}' and JSONs to '{OUTPUT_JSON_DIR}'")
