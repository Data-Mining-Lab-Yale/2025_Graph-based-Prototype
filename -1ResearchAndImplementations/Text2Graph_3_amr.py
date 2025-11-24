import os
import json
import penman
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from amrlib.models.parse_xfm.inference import Inference

# === Config ===
unit_level = "subsentence"  # "sentence" or "subsentence"
code_level = "subcode"      # "code", "subcode", or "subsubcode"

# Construct filename and output folders
input_filename = f"EPPC_output_json/{unit_level}_{code_level}_labels.json"
OUTPUT_IMG_DIR = Path(f"outputs/amr_graphs/{unit_level}_{code_level}/images")
OUTPUT_JSON_DIR = Path(f"outputs/amr_graphs/{unit_level}_{code_level}/json")
OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)

# AMR model path
model_dir = "Models/model_parse_xfm_bart_large"
tok_dir = "facebook/bart-large"
clean_frames = True  # Set False to keep "-01" frame suffixes

# === Load sentences from file ===
with open(input_filename, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# === Load AMR model ===
print("🔄 Loading AMR parser from local model path...")
stog = Inference(model_dir, model_fn="pytorch_model.bin", tok_name=tok_dir)

# === Process sentences ===
for key, entry in raw_data.items():
    sentence = entry["text"]
    print(f"🧾 Parsing {key}: {sentence}")
    try:
        amr_str = stog.parse_sents([sentence])[0]
        g = penman.decode(amr_str)
    except Exception as e:
        print(f"⚠️ Failed to parse {key}: {e}")
        continue

    # Clean and simplify labels
    nodes = []
    edges = []
    var_to_label = {}
    for src, role, tgt in g.triples:
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
    for t in g.variables():
        if t not in var_to_label:
            nodes.append({"id": t, "label": t})

    # === Save JSON ===
    graph = {"id": key, "text": sentence, "nodes": nodes, "edges": edges}
    json_path = OUTPUT_JSON_DIR / f"{key}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)
    print(f"✅ Saved JSON: {json_path}")

    # === Save Visualization ===
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
    plt.title(f"AMR Graph {key}")
    plt.tight_layout()
    img_path = OUTPUT_IMG_DIR / f"{key}.png"
    plt.savefig(img_path)
    plt.close()
    print(f"🖼️  Graph saved: {img_path}")
