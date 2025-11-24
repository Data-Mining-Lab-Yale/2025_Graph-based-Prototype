import json
import os
from pathlib import Path
import networkx as nx
from text_to_graph.Text2Graph_1_dependency import DependencyGraphBuilder
from text_to_graph.Text2Graph_1_visualize import visualize_graph

# === Config ===
unit_level = "subsentence"  # "sentence" or "subsentence"
code_level = "subcode"   # "code", "subcode", or "subsubcode"

# === Configuration ===
# INPUT = "sentence_interactional"
# INPUT = "sentence_goal_oriented"
# INPUT = "subsentence_interactional"
INPUT = "subsentence_goal_oriented"

# Construct filename and output folders
# input_filename = f"EPPC_output_json/{unit_level}_{code_level}_labels.json"
# OUTPUT_IMG_DIR = Path(f"outputs/dependency_graphs/{unit_level}_{code_level}/images")
# OUTPUT_JSON_DIR = Path(f"outputs/dependency_graphs/{unit_level}_{code_level}/json")
# OUTPUT_IMG_DIR = Path(f"outputs/dependency_graphs/{unit_level}_{code_level}/images")
# OUTPUT_JSON_DIR = Path(f"outputs/dependency_graphs/{unit_level}_{code_level}/json")
input_filename = f"EPPC_output_json/{INPUT}_label.json"
OUTPUT_IMG_DIR = Path(f"Dep_{INPUT}_graph/images")
OUTPUT_JSON_DIR = Path(f"Dep_{INPUT}_graph/json")
# OUTPUT_IMG_DIR = Path(f"outputs/dependency_graphs/{unit_level}_{code_level}/images")
# OUTPUT_JSON_DIR = Path(f"outputs/dependency_graphs/{unit_level}_{code_level}/json")
OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)

# === Load data ===
with open(input_filename, "r", encoding="utf-8") as f:
    data = json.load(f)

# === Build and save graphs ===
builder = DependencyGraphBuilder()

for i, (uid, entry) in enumerate(data.items()):
    text = entry["text"]
    if not text.strip():
        continue  # skip empty

    # Build graph
    G = builder.build_graph(text)

    # Save JSON
    json_path = OUTPUT_JSON_DIR / f"{uid}.json"
    with open(json_path, "w", encoding="utf-8") as f_json:
        json.dump(nx.node_link_data(G, edges="links"), f_json, indent=2)

    # Save PNG
    img_path = OUTPUT_IMG_DIR / f"{uid}.png"
    visualize_graph(
        G,
        title=f"{uid}",
        layout="spring",
        font_size=12,
        node_color="#a6cee3",
        output_path=str(img_path)
    )

print(f"✅ Finished generating {len(data)} graphs for {unit_level}-{code_level}.")
