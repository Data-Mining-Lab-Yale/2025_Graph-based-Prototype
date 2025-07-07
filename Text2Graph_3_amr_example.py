# AMR Parsing + Graph Visualization Script
import os
import json
import networkx as nx
import matplotlib.pyplot as plt
from amrlib import load_stog_model
import penman

# === Config ===
output_dir = "outputs/amr_graphs/"
os.makedirs(output_dir, exist_ok=True)
example_sentence = "Dr. Smith sent a PET scan order."

# === Load AMR model ===
print("Loading AMR parser...")
stog = load_stog_model()
graphs = stog.parse_sentences([example_sentence])
amr_str = graphs[0]

# === Save AMR string for reference ===
print("Parsed AMR:\n", amr_str)
with open(os.path.join(output_dir, "amr_penman.txt"), "w") as f:
    f.write(amr_str)

# === Convert to PENMAN graph ===
pm_graph = penman.decode(amr_str)

# === Build NetworkX graph ===
G = nx.DiGraph()
for triple in pm_graph.triples:
    source, role, target = triple
    G.add_node(source)
    G.add_node(target)
    G.add_edge(source, target, label=role)

# === Visualization ===
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, font_weight='bold', edge_color='gray')
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)
plt.title("AMR Graph for: " + example_sentence)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "amr_graph.png"))
plt.show()

# === Save as JSON structure ===
graph_data = {
    "sentence": example_sentence,
    "amr_penman": amr_str,
    "nodes": list(G.nodes),
    "edges": [
        {"source": u, "target": v, "label": G[u][v]["label"]}
        for u, v in G.edges
    ]
}
with open(os.path.join(output_dir, "amr_graph.json"), "w") as f:
    json.dump(graph_data, f, indent=2)

print("✅ Done. AMR graph and files saved to:", output_dir)
