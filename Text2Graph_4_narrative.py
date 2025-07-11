import os
import json
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

# === Config ===
unit_level = "subsentence"
code_level = "subcode"
graph_dir = Path(f"outputs/narrative_graphs/{unit_level}_{code_level}/json/")
output_img_dir = Path(f"outputs/narrative_graphs/{unit_level}_{code_level}/images/")
output_img_dir.mkdir(parents=True, exist_ok=True)

# === Optional: edge colors
edge_colors = {
    "next": "gray",
    "same_label": "blue",
    "contrast": "red",
    "elaboration": "green"
}

# === Visualize each graph ===
for file in graph_dir.glob("*.json"):
    with open(file, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    G = nx.DiGraph()
    labels = {}

    for node in graph_data["nodes"]:
        G.add_node(node["id"])
        labels[node["id"]] = node["label"]

    for edge in graph_data["edges"]:
        G.add_edge(edge["source"], edge["target"], label=edge["label"])

    pos = nx.spring_layout(G, seed=42)
    edge_labels = nx.get_edge_attributes(G, "label")
    edge_color_vals = [edge_colors.get(lbl, "black") for lbl in edge_labels.values()]

    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=2500, node_color="lightblue", font_size=7,
            arrows=True, edge_color=edge_color_vals)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
    plt.title(f"Narrative Graph: {graph_data['message_id']}")
    plt.tight_layout()
    plt.savefig(output_img_dir / f"{graph_data['message_id']}.png")
    plt.close()
