import os
import json
import networkx as nx
import matplotlib.pyplot as plt
from transformers import pipeline
from pathlib import Path

# === Config ===
unit_level = "subsentence"  # "sentence" or "subsentence"
code_level = "subcode"      # "code", "subcode", or "subsubcode"
input_json_path = f"EPPC_output_json/{unit_level}_{code_level}_labels.json"
output_dir = Path(f"outputs/srl_graphs_weighted/{unit_level}_{code_level}/")
output_dir.mkdir(parents=True, exist_ok=True)

# === Load classifier ===
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# === Define canonical semantic roles ===
canonical_roles = ["agent", "action", "time", "location", "recipient"]

# === Load input ===
with open(input_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# === Process each item ===
for idx, (sid, item) in enumerate(data.items()):
    sentence = item["text"]

    # 1. Run classification
    result = classifier(sentence, candidate_labels=canonical_roles)

    # 2. Build graph
    G = nx.DiGraph()
    G.add_node("Sentence", label=sentence)

    for label, score in zip(result["labels"], result["scores"]):
        G.add_node(label, score=score)
        G.add_edge("Sentence", label, weight=score)

    # 3. Save graph JSON
    graph_data = {
        "sentence_id": sid,
        "text": sentence,
        "graph_type": "semantic_proxy_weighted",
        "nodes": [{"id": label, "type": "role", "score": float(f"{score:.4f}")} for label, score in zip(result["labels"], result["scores"])] + [{"id": "Sentence", "type": "text", "label": sentence}],
        "edges": [{"source": "Sentence", "target": label, "weight": float(f"{score:.4f}")} for label, score in zip(result["labels"], result["scores"])]
    }
    with open(output_dir / f"{sid}.json", "w", encoding="utf-8") as f_out:
        json.dump(graph_data, f_out, indent=2)

    # 4. Save visualization
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G, seed=42)
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}

    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="#A7C7E7", font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title(f"Weighted Role Graph: {sid}")
    plt.tight_layout()
    plt.savefig(output_dir / f"{sid}.png")
    plt.close()

print(f"Done. Graphs saved in {output_dir}")
