import json
import networkx as nx
import matplotlib.pyplot as plt
import os
from pathlib import Path

# === Settings ===
INPUT_FILE = "EPPC_output_json/messages_with_sentences_and_subsentences.json"
OUTPUT_DIR = "EPPC_output_json/structure_visualized_graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    # Add message node
    msg_id = message["message_id"]
    G.add_node(msg_id, label="message", text=message["message"])

    # Add sentence and subsentence nodes
    for sent in message["sentences"]:
        sent_id = sent["sentence_id"]
        G.add_node(sent_id, label="sentence", text=sent["sentence"])
        G.add_edge(msg_id, sent_id)

        for sub in sent["subsentences"]:
            sub_id = sub["subsentence_id"]
            G.add_node(sub_id, label="subsentence", text=sub["subsentence"])
            G.add_edge(sent_id, sub_id)

    # Layout and Labels
    pos = hierarchical_layout(G, root=msg_id)
    labels = {n: f'{d["label"]}: {d["text"][:30]}...' for n, d in G.nodes(data=True)}

    # Draw and Save
    plt.figure(figsize=(14, 10))
    nx.draw(G, pos, with_labels=True, labels=labels,
            node_size=2500, node_color="lightyellow",
            font_size=8, edge_color="gray", arrows=True)
    plt.title(f"Message Structure: {msg_id}")
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f"message_{i+1:03d}_{msg_id}.png")
    plt.savefig(output_path)
    plt.close()

print(f"Saved {len(data)} visualizations to '{OUTPUT_DIR}'")
