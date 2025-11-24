import os
import json
import torch
from pathlib import Path
from tqdm import tqdm
from torch_geometric.data import Data

# === Paths ===
INPUT_DIR = Path("filtered_dependency_graphs/")
LABEL_MAPPING_FILE = "EPPC_output_json/annotation_code_mapping_detailed_corrected.json"
OUTPUT_FILE = "dep_graph_features.pt"

# === Load label mapping ===
with open(LABEL_MAPPING_FILE, "r") as f:
    raw2canonical = json.load(f)

canonical_labels = sorted(set(v["matched_codebook_label"] for v in raw2canonical.values()))
IDX2LABEL = canonical_labels
LABEL2IDX = {label: i for i, label in enumerate(IDX2LABEL)}

def get_label_index(raw_label):
    try:
        canonical = raw2canonical[raw_label]["matched_codebook_label"]
        return LABEL2IDX[canonical]
    except:
        return None

# === Process ===
dataset = []
fail_count = 0

for file in tqdm(list(INPUT_DIR.glob("*.json"))):
    try:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        msg_id = file.stem
        raw_label = data.get("label", "")
        text = data.get("text", "")
        label_idx = get_label_index(raw_label)
        if label_idx is None:
            print(f"⚠️ Unknown label in {file.name}: {raw_label}")
            continue

        nodes = data.get("nodes", [])
        links = data.get("links", [])

        if not nodes or not links:
            continue

        # Node features: pos tag embedding (optional: add 'label' later)
        pos_tags = [node["pos"] for node in nodes]
        pos_vocab = {tag: i for i, tag in enumerate(sorted(set(pos_tags)))}
        x = torch.eye(len(pos_tags))

        # Edges
        edge_index = torch.tensor([[link["source"], link["target"]] for link in links], dtype=torch.long).t().contiguous()

        # One node per token, one subsentence per file
        graph = Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor([label_idx], dtype=torch.long),
            label=IDX2LABEL[label_idx],
            text=text,
            message_id=msg_id,
            subsentence_index=0
        )
        dataset.append(graph)

    except Exception as e:
        print(f"⚠️ Failed on {file.name}: {e}")
        fail_count += 1

# === Save ===
print(f"✅ Saved {len(dataset)} graphs to: {OUTPUT_FILE}")
torch.save(dataset, OUTPUT_FILE)
