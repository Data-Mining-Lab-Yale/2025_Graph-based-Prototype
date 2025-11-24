import os
import json
import networkx as nx
import matplotlib.pyplot as plt
from allennlp.predictors.predictor import Predictor

# 1. Load model
print("Loading AllenNLP SRL model...")
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.11.19.tar.gz")

# 2. Example input sentence
sentence_id = "test_0"
sentence_text = "Dr. Smith sent a PET scan order."

# 3. Run prediction
output = predictor.predict(sentence=sentence_text)

# 4. Build semantic graph
G = nx.DiGraph()
G.graph["graph_type"] = "semantic"
G.graph["text"] = sentence_text
G.graph["sentence_id"] = sentence_id

# 5. For each verb/predicate in sentence
for i, verb in enumerate(output["verbs"]):
    tags = verb["tags"]
    words = output["words"]
    predicate = verb["verb"]

    pred_node = f"p{i}"
    G.add_node(pred_node, label=predicate, type="predicate")

    current_arg = []
    current_role = None
    for word, tag in zip(words, tags):
        if tag.startswith("B-"):
            if current_arg and current_role:
                arg_text = " ".join(current_arg)
                arg_id = f"a{i}_{current_role}"
                G.add_node(arg_id, label=arg_text, type="argument", role=current_role)
                G.add_edge(pred_node, arg_id, label=current_role)
            current_role = tag[2:]
            current_arg = [word]
        elif tag.startswith("I-") and current_role:
            current_arg.append(word)
        else:
            if current_arg and current_role:
                arg_text = " ".join(current_arg)
                arg_id = f"a{i}_{current_role}"
                G.add_node(arg_id, label=arg_text, type="argument", role=current_role)
                G.add_edge(pred_node, arg_id, label=current_role)
            current_arg = []
            current_role = None

    if current_arg and current_role:
        arg_text = " ".join(current_arg)
        arg_id = f"a{i}_{current_role}"
        G.add_node(arg_id, label=arg_text, type="argument", role=current_role)
        G.add_edge(pred_node, arg_id, label=current_role)

# 6. Save graph as JSON
os.makedirs("semantic_graphs", exist_ok=True)
json_path = os.path.join("semantic_graphs", f"{sentence_id}.json")

graph_data = {
    "graph_type": G.graph["graph_type"],
    "text": G.graph["text"],
    "sentence_id": G.graph["sentence_id"],
    "nodes": [
        {"id": n, **G.nodes[n]} for n in G.nodes()
    ],
    "edges": [
        {"source": u, "target": v, "label": d["label"]} for u, v, d in G.edges(data=True)
    ]
}

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(graph_data, f, indent=2)
print(f"Graph saved to {json_path}")

# 7. (Optional) Visualize
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G, seed=42)
labels = nx.get_node_attributes(G, "label")
nx.draw(G, pos, with_labels=True, labels=labels, node_color="lightblue", node_size=1500, font_size=10, edge_color="gray")
edge_labels = nx.get_edge_attributes(G, "label")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)
plt.title(f"Semantic Graph for: {sentence_text}")
plt.tight_layout()
plt.savefig(os.path.join("semantic_graphs", f"{sentence_id}.png"))
plt.show()
