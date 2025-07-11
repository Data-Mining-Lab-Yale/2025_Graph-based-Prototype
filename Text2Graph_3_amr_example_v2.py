import os
import penman
import json
import matplotlib.pyplot as plt
import networkx as nx
from amrlib.models.parse_xfm.inference import Inference

# === Config ===
model_dir = "Models/model_parse_xfm_bart_large"
tok_dir = "facebook/bart-large"
output_dir = "outputs/amr_graph_example/"
os.makedirs(output_dir, exist_ok=True)

INPUT_SENTENCES = [
    "Dr. Smith sent a PET scan order.",
    "The patient was scheduled for a follow-up next Monday."
]
clean_frames = True  # Set False to keep "-01" frame suffixes

# === Load model ===
print("🔄 Loading AMR parser from local model path...")
stog = Inference(model_dir, model_fn="pytorch_model.bin", tok_name=tok_dir)

# === Processing ===
for idx, sentence in enumerate(INPUT_SENTENCES):
    print(f"🧾 Parsing: {sentence}")
    amr_str = stog.parse_sents([sentence])[0]
    g = penman.decode(amr_str)

    # Clean and simplify labels
    nodes = []
    edges = []
    var_to_label = {}
    for triple in g.triples:
        src, role, tgt = triple
        if role == ":instance":
            label = tgt
            if clean_frames and label.endswith("-01"):
                label = label[:-3]
            if label.isnumeric():
                label = f"Num:{label}"
            var_to_label[src] = label
            nodes.append({"id": src, "label": label})
        else:
            edges.append({"source": src, "target": tgt, "label": role})

    # Ensure all variables are represented
    for t in g.variables():  # ← FIXED HERE
        if t not in var_to_label:
            nodes.append({"id": t, "label": t})

    # Save JSON
    graph = {"text": sentence, "nodes": nodes, "edges": edges}
    json_path = os.path.join(output_dir, f"amr_{idx}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)
    print(f"✅ Saved: {json_path}")

    # Visualize
    G = nx.DiGraph()
    for node in nodes:
        G.add_node(node["id"], label=node["label"])
    for edge in edges:
        G.add_edge(edge["source"], edge["target"], label=edge["label"])

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, "label"),
            node_size=2500, node_color="lightblue", font_size=8, arrows=True)
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    plt.title(f"AMR Graph {idx}")
    plt.tight_layout()
    img_path = os.path.join(output_dir, f"amr_{idx}.png")
    plt.savefig(img_path)
    plt.close()
    print(f"🖼️  Graph saved: {img_path}")
