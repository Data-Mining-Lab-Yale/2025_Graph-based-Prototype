# run_pipeline.py
# from text_to_graph.dependency import DependencyGraphBuilder
# from text_to_graph.visualize import visualize_graph

from text_to_graph.Text2Graph_1_dependency import DependencyGraphBuilder
from text_to_graph.Text2Graph_1_visualize import visualize_graph
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# if __name__ == "__main__":
#     text = "The doctor scheduled an MRI for the patient next Monday."

#     builder = DependencyGraphBuilder()
#     G = builder.build_graph(text)

#     visualize_graph(G, title="Dependency Graph")


import json
from text_to_graph.Text2Graph_1_dependency import DependencyGraphBuilder
from text_to_graph.Text2Graph_1_visualize import visualize_graph

# === Config ===
EXAMPLE_ID = "3_0_2"  # Change this to test a different subsentence
LABEL_FILE = "EPPC_output_json/subsentence_subsubcode_labels.json"

# === Load text from dataset ===
with open(LABEL_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

if EXAMPLE_ID not in data:
    raise ValueError(f"Subsentence ID {EXAMPLE_ID} not found in dataset.")

text = data[EXAMPLE_ID]["text"]

# === Build dependency graph ===
builder = DependencyGraphBuilder()
G = builder.build_graph(text)

# === Visualize ===
print(f"Visualizing subsentence: {text}")
visualize_graph(
    G,
    title=f"Dependency Graph for {EXAMPLE_ID}",
    layout="spring",
    font_size=12,
    node_color="#a6cee3"
)
