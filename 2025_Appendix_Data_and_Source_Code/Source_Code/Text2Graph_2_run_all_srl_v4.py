import os
import json
import matplotlib.pyplot as plt
import networkx as nx
from allennlp.predictors.predictor import Predictor
import allennlp_models.structured_prediction

# === Config ===
unit_level = "subsentence"  # "sentence" or "subsentence"
code_level = "subcode"      # "code", "subcode", or "subsubcode"

# Input and Output Paths
input_json_path = f"EPPC_output_json/{unit_level}_{code_level}_labels.json"
output_dir = f"outputs/srl_graphs_more_role/{unit_level}_{code_level}/"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "json"), exist_ok=True)

# Load SRL Predictor
print("Loading AllenNLP SRL model...")
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.11.19.tar.gz")

# Richer descriptions for semantic roles
role_descriptions = {
    "ARG0": "Agent (doer of the action)", "ARG1": "Theme (thing affected by the action)",
    "ARG2": "Beneficiary/instrument/end state", "ARG3": "Starting point", "ARG4": "Destination/end point",
    "ARGM-TMP": "Temporal (when)", "ARGM-LOC": "Location (where)", "ARGM-MNR": "Manner (how)",
    "ARGM-CAU": "Cause (why)", "ARGM-DIR": "Direction", "ARGM-EXT": "Extent", "ARGM-NEG": "Negation",
    "ARGM-MOD": "Modal", "ARGM-PRP": "Purpose", "ARGM-ADV": "Adverbial", "ARGM-REC": "Reciprocal",
    "ARGM-COM": "Comitative (with whom)", "ARGM-DIS": "Discourse marker", "ARGM-PRD": "Secondary predicate",
    "ARGM-GOL": "Goal", "ARGM-LVB": "Light verb", "V": "Predicate (verb)"
}

# Load input data
with open(input_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Process each sentence
for sid, entry in data.items():
    sentence = entry["text"]
    output = predictor.predict(sentence=sentence)

    graph_data = {
        "graph_type": "semantic_proxy",
        "sentence_id": sid,
        "text": sentence,
        "nodes": [],
        "edges": []
    }

    G = nx.DiGraph()
    center_id = "center"
    graph_data["nodes"].append({"id": center_id, "label": sentence, "type": unit_level})
    G.add_node(center_id, label=sentence, color='lightblue')

    predicate_count = 0
    for verb in output.get("verbs", []):
        tags = verb["tags"]
        words = output["words"]
        predicate_word = verb["verb"]
        pred_id = f"p{predicate_count}"
        predicate_count += 1

        G.add_node(pred_id, label=predicate_word, color="orange")
        graph_data["nodes"].append({
            "id": pred_id, "label": predicate_word, "type": "predicate",
            "role": "V", "description": role_descriptions.get("V", "")
        })
        G.add_edge(center_id, pred_id, label="predicate")
        graph_data["edges"].append({"source": center_id, "target": pred_id, "label": "predicate"})

        current_role = None
        current_words = []
        for word, tag in zip(words, tags):
            if tag.startswith("B-"):
                if current_role and current_words:
                    span = " ".join(current_words)
                    node_id = f"{current_role}_{span[:10]}"
                    G.add_node(node_id, label=span, color="lightgreen")
                    G.add_edge(pred_id, node_id, label=current_role)
                    graph_data["nodes"].append({
                        "id": node_id, "label": span, "type": "argument",
                        "role": current_role, "description": role_descriptions.get(current_role, "")
                    })
                    graph_data["edges"].append({
                        "source": pred_id, "target": node_id, "label": current_role
                    })
                current_role = tag[2:]
                current_words = [word]
            elif tag.startswith("I-") and current_role:
                current_words.append(word)
            elif tag == "O" and current_role:
                span = " ".join(current_words)
                node_id = f"{current_role}_{span[:10]}"
                G.add_node(node_id, label=span, color="lightgreen")
                G.add_edge(pred_id, node_id, label=current_role)
                graph_data["nodes"].append({
                    "id": node_id, "label": span, "type": "argument",
                    "role": current_role, "description": role_descriptions.get(current_role, "")
                })
                graph_data["edges"].append({
                    "source": pred_id, "target": node_id, "label": current_role
                })
                current_role = None
                current_words = []

    # Save results
    json_path = os.path.join(output_dir, "json", f"{sid}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2)

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True,
            node_color=[G.nodes[n].get("color", "gray") for n in G.nodes],
            node_size=2000, font_size=10, font_weight="bold", arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(e[0], e[1]): e[2]["label"] for e in G.edges(data=True)}, font_color='red')
    plt.title(f"SRL Graph for {sid}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "images", f"{sid}.png"))
    plt.close()
