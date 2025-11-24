import penman
import networkx as nx
import matplotlib.pyplot as plt
import json
from pathlib import Path
from amrlib.models.parse_xfm.inference import Inference

# === Config ===
model_dir = "Models/model_parse_xfm_bart_large"
tok_dir = "facebook/bart-large"
sentence = "Dr. Smith sent a PET scan order."
output_json_path = "output_amr_graph_cleaned.json"
output_img_path = "output_amr_graph_cleaned.png"

# === Load AMR parser ===
print("🔄 Loading AMR parser from local model path...")
stog = Inference(model_dir, model_fn="pytorch_model.bin", tok_name=tok_dir)

# === Parse sentence ===
print("🧾 Parsing sentence:", sentence)
amr_strings = stog.parse_sents([sentence])
amr_graph = penman.decode(amr_strings[0])

# === Remap node ids to concepts ===
concept_map = {}
for triple in amr_graph.triples:
    if triple[1] == ":instance":
        concept = triple[2]
        count = sum(1 for c in concept_map.values() if c.startswith(concept))
        new_id = concept if count == 0 else f"{concept}_{count+1}"
        concept_map[triple[0]] = new_id

# === Build graph data ===
nodes = list(concept_map.values())
edges = []
for source, role, target in amr_graph.triples:
    if role == ":instance":
        continue
    if source in concept_map and target in concept_map:
        edges.append({"source": concept_map[source], "target": concept_map[target], "label": role})

graph_dict = {
    "graph_type": "amr_cleaned",
    "text": sentence,
    "nodes": [{"id": n, "label": n} for n in nodes],
    "edges": edges
}

# === Save JSON ===
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(graph_dict, f, indent=2)
print(f"✅ Graph saved to {output_json_path}")

# === Visualize ===
G = nx.DiGraph()
for node in graph_dict["nodes"]:
    G.add_node(node["id"], label=node["label"])
for edge in graph_dict["edges"]:
    G.add_edge(edge["source"], edge["target"], label=edge["label"])

pos = nx.spring_layout(G, seed=42, k=1.5)
plt.figure(figsize=(10, 7))
nx.draw_networkx_nodes(G, pos, node_size=1500, node_color='lightblue', edgecolors='black')
nx.draw_networkx_labels(G, pos, font_size=10)
nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=15)
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d["label"] for u, v, d in G.edges(data=True)}, font_color='green')

plt.axis('off')
plt.tight_layout()
plt.savefig(output_img_path)
print(f"🖼️ Graph image saved to {output_img_path}")
plt.show()
