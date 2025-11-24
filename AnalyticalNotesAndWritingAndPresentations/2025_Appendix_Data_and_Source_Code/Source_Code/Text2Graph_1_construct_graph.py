import json
import os
from pathlib import Path
from text_to_graph.Text2Graph_1_base import GraphBuilder
from text_to_graph.Text2Graph_1_visualize import visualize_graph
import networkx as nx

import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# === Config ===
LABEL_LEVEL = "subsentence"  # or "sentence"
LABEL_FILE = f"EPPC_output_json/{LABEL_LEVEL}_subcode_labels.json"
OUTPUT_GRAPH_DIR = Path("outputs/text2graphs_subsetence_order_json")
OUTPUT_IMG_DIR = Path("outputs/text2graphs_subsetence_order_viz")

OUTPUT_GRAPH_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

# === Load data ===
with open(LABEL_FILE, "r") as f:
    data = json.load(f)

# === Group by message_id ===
grouped = {}
for uid, entry in data.items():
    message_id = uid.split("_")[0]
    if message_id not in grouped:
        grouped[message_id] = []
    grouped[message_id].append((uid, entry))

# === Construct and save ===
for msg_id, entries in grouped.items():
    G = nx.DiGraph()
    for uid, entry in sorted(entries):
        node_id = uid
        G.add_node(node_id, text=entry["text"], label=entry["labels"][0]["label"])

    # Add sequential edges
    sorted_entries = sorted(entries)
    for i in range(len(sorted_entries) - 1):
        G.add_edge(sorted_entries[i][0], sorted_entries[i + 1][0], type="next")

    # Save graph JSON
    json_path = OUTPUT_GRAPH_DIR / f"graph_{LABEL_LEVEL}_{msg_id}.json"
    with open(json_path, "w") as f:
        json.dump(nx.node_link_data(G), f, indent=2)

    # Save visualization
    png_path = OUTPUT_IMG_DIR / f"graph_{LABEL_LEVEL}_{msg_id}.png"
    visualize_graph(G, output_path=str(png_path), layout="spring", font_size=14, node_color="#a6cee3")
