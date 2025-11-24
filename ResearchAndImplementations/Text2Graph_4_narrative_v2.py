import json
import os
from pathlib import Path
from collections import defaultdict

# === Config ===
unit_level = "subsentence"
code_level = "subcode"
input_path = f"EPPC_output_json/{unit_level}_{code_level}_labels.json"
output_dir = Path(f"outputs/narrative_weighted_graphs/{unit_level}_{code_level}/")
output_dir.mkdir(parents=True, exist_ok=True)

# === Customizable edge weights ===
relation_weights = {
    "next": 0.8,
    "same_label": 0.7,
    "elaboration": 0.6,
    "contrast": 0.5,
    "self": 1.0
}
use_discourse_edges = False  # Optional: enable cue-based edges later

cue_keywords = {
    "contrast": {"but", "however", "although"},
    "elaboration": {"because", "so", "since", "therefore"}
}

# === Load data ===
with open(input_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# === Group by message ID ===
messages = defaultdict(list)
for clause_id, entry in raw_data.items():
    msg_id = clause_id.split("_")[0]
    text = entry["text"]
    code = entry.get("labels", [{}])[0].get("label", "Unknown")
    messages[msg_id].append({"id": clause_id, "text": text, "code": code})

# === Generate a centered graph for each clause ===
for msg_id, clauses in messages.items():
    clauses.sort(key=lambda x: x["id"])

    # Index all clauses for fast lookup
    id_to_clause = {c["id"]: c for c in clauses}

    for center in clauses:
        cid = center["id"]
        graph = {
            "center_id": cid,
            "nodes": [],
            "edges": []
        }

        # Include all nodes in message (including center)
        for c in clauses:
            graph["nodes"].append({"id": c["id"], "label": c["text"]})

        # Self-loop (or implicit node weight)
        graph["edges"].append({
            "source": cid,
            "target": cid,
            "label": "self",
            "weight": relation_weights["self"]
        })

        # Temporal edges
        for i in range(len(clauses) - 1):
            src, tgt = clauses[i]["id"], clauses[i + 1]["id"]
            if cid in [src, tgt]:  # only keep edges relevant to center
                graph["edges"].append({
                    "source": src,
                    "target": tgt,
                    "label": "next",
                    "weight": relation_weights["next"]
                })

        # Same-label edges
        for other in clauses:
            if other["id"] != cid and other["code"] == center["code"]:
                graph["edges"].append({
                    "source": cid,
                    "target": other["id"],
                    "label": "same_label",
                    "weight": relation_weights["same_label"]
                })

        # Optional: Cue-based edges
        if use_discourse_edges:
            text_lower = center["text"].lower()
            for cue_type, keywords in cue_keywords.items():
                if any(kw in text_lower for kw in keywords):
                    for other in clauses:
                        if other["id"] != cid:
                            graph["edges"].append({
                                "source": cid,
                                "target": other["id"],
                                "label": cue_type,
                                "weight": relation_weights[cue_type]
                            })

        # Save graph to individual file
        with open(output_dir / f"{cid}.json", "w", encoding="utf-8") as f:
            json.dump(graph, f, indent=2)
