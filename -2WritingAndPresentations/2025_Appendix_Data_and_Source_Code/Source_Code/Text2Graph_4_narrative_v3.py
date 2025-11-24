import json
from pathlib import Path
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt

# === Config ===
unit_level = "subsentence"
code_level = "subcode"
input_path = Path(f"EPPC_output_json/{unit_level}_{code_level}_labels.json")
base_output_dir = Path(f"outputs/narrative_egograph_per_clause/{unit_level}_{code_level}/")
json_dir = base_output_dir / "json"
img_dir = base_output_dir / "images"
json_dir.mkdir(parents=True, exist_ok=True)
img_dir.mkdir(parents=True, exist_ok=True)

# === Edge weights
relation_weights = {
    "next": 0.8,
    "prev": 0.8,
    "same_label": 0.7,
    "elaboration": 0.6,
    "contrast": 0.5,
    "self": 1.0
}

edge_colors = {
    "next": "gray",
    "prev": "gray",
    "same_label": "blue",
    "elaboration": "green",
    "contrast": "red",
    "self": "black"
}

cue_keywords = {
    "contrast": {"but", "however", "although", "yet", "whereas"},
    "elaboration": {"because", "so", "since", "therefore", "in order to", "as a result"}
}

# === Load annotations
with open(input_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# === Group by message ID
messages = defaultdict(list)
for cid, entry in raw_data.items():
    msg_id = cid.split("_")[0]
    messages[msg_id].append({
        "id": cid,
        "text": entry["text"],
        "code": entry.get("labels", [{}])[0].get("label", "Unknown")
    })

# === Generate ego-graph for each clause
for msg_id, clauses in messages.items():
    clauses.sort(key=lambda x: x["id"])
    id_to_text = {c["id"]: c["text"].lower() for c in clauses}
    id_to_code = {c["id"]: c["code"] for c in clauses}  # ✅ Fixed here
    all_ids = [c["id"] for c in clauses]

    for center_idx, center in enumerate(clauses):
        cid = center["id"]
        center_code = center["code"]
        center_text = center["text"].lower()

        nodes = [{"id": cid, "label": center["text"]}]
        edges = [{"source": cid, "target": cid, "label": "self", "weight": relation_weights["self"]}]

        # Add next
        if center_idx < len(clauses) - 1:
            next_id = clauses[center_idx + 1]["id"]
            nodes.append({"id": next_id, "label": clauses[center_idx + 1]["text"]})
            edges.append({
                "source": cid, "target": next_id, "label": "next", "weight": relation_weights["next"]
            })

        # Add prev
        if center_idx > 0:
            prev_id = clauses[center_idx - 1]["id"]
            nodes.append({"id": prev_id, "label": clauses[center_idx - 1]["text"]})
            edges.append({
                "source": prev_id, "target": cid, "label": "prev", "weight": relation_weights["prev"]
            })

        # Add same_label
        for other in clauses:
            if other["id"] != cid and other["code"] == center_code:
                if all(n["id"] != other["id"] for n in nodes):
                    nodes.append({"id": other["id"], "label": other["text"]})
                edges.append({
                    "source": cid, "target": other["id"], "label": "same_label", "weight": relation_weights["same_label"]
                })

        # Cue-based
        for cue_type, keywords in cue_keywords.items():
            if any(kw in center_text for kw in keywords):
                if center_idx + 1 < len(clauses):
                    target_id = clauses[center_idx + 1]["id"]
                    if all(n["id"] != target_id for n in nodes):
                        nodes.append({"id": target_id, "label": id_to_text[target_id]})
                    edges.append({
                        "source": cid, "target": target_id,
                        "label": cue_type, "weight": relation_weights[cue_type]
                    })

        # === Save JSON
        graph = {
            "center_id": cid,
            "message_id": msg_id,
            "nodes": nodes,
            "edges": edges
        }

        with open(json_dir / f"{cid}.json", "w", encoding="utf-8") as f:
            json.dump(graph, f, indent=2)

        # === Visualization
        G = nx.DiGraph()
        node_labels = {n["id"]: n["label"] for n in nodes}
        edge_labels = {}
        edge_weights = []
        edge_colors_used = []

        for edge in edges:
            G.add_edge(edge["source"], edge["target"])
            edge_labels[(edge["source"], edge["target"])] = edge["label"]
            edge_weights.append(edge["weight"] * 2.0)
            edge_colors_used.append(edge_colors.get(edge["label"], "black"))

        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(10, 6))
        nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="lightblue")
        nx.draw_networkx_labels(G, pos, node_labels, font_size=7)
        nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color=edge_colors_used, arrows=True)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
        plt.title(f"Ego-Graph: {cid}")
        plt.tight_layout()
        plt.savefig(img_dir / f"{cid}.png")
        plt.close()

print("✅ All ego-graphs with corrected next/prev and organized folders saved.")
