import os
import json
from allennlp.predictors.predictor import Predictor
import matplotlib.pyplot as plt
import networkx as nx

# ---- SETTINGS ----
# === Config ===
unit_level = "subsentence"  # "sentence" or "subsentence"
code_level = "subcode"   # "code", "subcode", or "subsubcode"

# Construct filename and output folders
input_json_path = f"EPPC_output_json/{unit_level}_{code_level}_labels.json"
# OUTPUT_IMG_DIR = Path(f"outputs/dependency_graphs/{unit_level}_{code_level}/images")
# OUTPUT_JSON_DIR = Path(f"outputs/dependency_graphs/{unit_level}_{code_level}/json")

# input_json_path = "sentence_code_labels.json"  # change to your file
output_dir = f"outputs/srl_graphs/{unit_level}_{code_level}/"
os.makedirs(output_dir, exist_ok=True)

# ---- Load model ----
print("Loading AllenNLP SRL model...")
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.11.19.tar.gz")

# ---- Role definitions ----
role_descriptions = {
    "ARG0": "Agent (doer of the action)",
    "ARG1": "Theme (thing affected by the action)",
    "ARG2": "Beneficiary, instrument, or end state",
    "ARG3": "Starting point",
    "ARG4": "Destination or end point",
    "ARGM-TMP": "Temporal modifier (when)",
    "ARGM-LOC": "Locative modifier (where)",
    "ARGM-MNR": "Manner modifier (how)",
    "ARGM-CAU": "Cause (why)",
    "ARGM-DIR": "Direction",
    "ARGM-EXT": "Extent",
    "ARGM-NEG": "Negation",
    "ARGM-MOD": "Modal verb",
    "ARGM-PRP": "Purpose",
    "ARGM-ADV": "Adverbial modifier",
    "ARGM-REC": "Reciprocals",
    "ARGM-COM": "Comitative (with whom)",
    "ARGM-DIS": "Discourse marker",
    "ARGM-PRD": "Secondary predicate",
    "ARGM-GOL": "Goal",
    "ARGM-LVB": "Light verb",
    "V": "Verb (predicate)"
}

# ---- Load input ----
with open(input_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# ---- Process each entry ----
for sentence_id, entry in data.items():
    sentence = entry["text"]
    print(f"\nProcessing {sentence_id}: {sentence}")

    try:
        result = predictor.predict(sentence=sentence)
    except Exception as e:
        print(f"Failed SRL for {sentence_id}: {e}")
        continue

    words = result["words"]

    for i, verb in enumerate(result["verbs"]):
        tags = verb["tags"]
        roles = {}
        current_role = None
        span_tokens = []

        for tag, word in zip(tags, words):
            if tag.startswith("B-"):
                if current_role:
                    roles[current_role] = " ".join(span_tokens)
                current_role = tag[2:]
                span_tokens = [word]
            elif tag.startswith("I-") and current_role:
                span_tokens.append(word)
            elif tag == "O":
                if current_role:
                    roles[current_role] = " ".join(span_tokens)
                    current_role = None
                    span_tokens = []

        if current_role and span_tokens:
            roles[current_role] = " ".join(span_tokens)

        # ---- Graph structure ----
        pred_node_id = "n0"
        graph_json = {
            "graph_type": "semantic_srl",
            "sentence_id": sentence_id,
            "text": sentence,
            "predicate": roles.get("V", verb["verb"]),
            "nodes": [],
            "edges": []
        }

        G = nx.DiGraph()
        G.add_node(pred_node_id, label=roles.get("V", verb["verb"]))

        graph_json["nodes"].append({
            "id": pred_node_id,
            "label": roles.get("V", verb["verb"]),
            "type": "predicate",
            "role": "V",
            "description": role_descriptions.get("V", "")
        })

        for j, (role, span) in enumerate(roles.items()):
            if role == "V":
                continue
            node_id = f"n{j+1}"
            G.add_node(node_id, label=span)
            G.add_edge(pred_node_id, node_id, label=role)

            graph_json["nodes"].append({
                "id": node_id,
                "label": span,
                "type": "argument",
                "role": role,
                "description": role_descriptions.get(role, "Unknown role")
            })
            graph_json["edges"].append({
                "source": pred_node_id,
                "target": node_id,
                "label": role
            })

        # ---- Save graph JSON ----
        json_path = os.path.join(output_dir, f"{sentence_id}_srl.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(graph_json, f, indent=2)
        print(f"Saved JSON: {json_path}")

        # ---- Save graph image ----
        pos = nx.spring_layout(G)
        edge_labels = nx.get_edge_attributes(G, 'label')
        node_labels = nx.get_node_attributes(G, 'label')

        plt.figure(figsize=(10, 6))
        nx.draw(G, pos, labels=node_labels, node_color='lightblue', node_size=2000, font_size=10, edgecolors='black')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='darkgreen')
        plt.title(f"SRL Graph - {sentence_id}")
        plt.tight_layout()

        img_path = os.path.join(output_dir, f"{sentence_id}_srl.png")
        plt.savefig(img_path)
        plt.close()
        print(f"Saved image: {img_path}")
