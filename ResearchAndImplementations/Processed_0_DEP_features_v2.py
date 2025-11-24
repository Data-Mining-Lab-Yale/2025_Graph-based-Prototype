import os
import json
import torch
from tqdm import tqdm
from pathlib import Path
from torch_geometric.data import Data
from collections import defaultdict

# === Config ===
INPUT_DIR = Path("filtered_dependency_graphs")
OUTPUT_FILE = "dep_graph_features.pt"
LOG_FILE = "dep_graph_fail_log.txt"

# === First pass: build global POS tag vocabulary ===
pos_vocab = set()
files = sorted(INPUT_DIR.glob("*.json"))
for file in tqdm(files, desc="Building POS tag vocab"):
    try:
        with open(file, "r", encoding="utf-8") as f:
            graph = json.load(f)
        for node in graph["nodes"]:
            pos_vocab.add(node["pos"])
    except Exception:
        continue

pos_list = sorted(pos_vocab)
pos2idx = {pos: i for i, pos in enumerate(pos_list)}
pos_dim = len(pos2idx)

# === Second pass: convert each graph ===
data_list = []
fail_log = []

for file in tqdm(files, desc="Converting graphs"):
    try:
        with open(file, "r", encoding="utf-8") as f:
            graph = json.load(f)

        # Node features (POS one-hot)
        x = torch.zeros((len(graph["nodes"]), pos_dim))
        for node in graph["nodes"]:
            idx = node["id"]
            pos = node["pos"]
            if pos in pos2idx:
                x[idx][pos2idx[pos]] = 1.0

        # Edge index
        edge_index = torch.tensor([[link["source"], link["target"]] for link in graph["links"]], dtype=torch.long).t().contiguous()

        # Label
        label_str = graph["label"]
        label_idx = -1  # will encode later if needed
        y = torch.tensor([label_idx], dtype=torch.long)

        # Text
        text = graph.get("text", "")

        # File name → message ID
        stem = file.stem  # e.g., 0_1_0
        message_id = stem
        subsentence_index = 0  # always 0 for dependency input (1 subsentence per file)

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            label=label_str,
            text=text,
            message_id=message_id,
            subsentence_index=subsentence_index,
        )
        data_list.append(data)

    except Exception as e:
        fail_log.append(f"⚠️ Failed on {file.name}: {str(e)}")

# === Save ===
torch.save(data_list, OUTPUT_FILE)
print(f"✅ Saved {len(data_list)} graphs to: {OUTPUT_FILE}")

# === Log any failures ===
if fail_log:
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        for line in fail_log:
            f.write(line + "\n")
    print(f"⚠️ Wrote {len(fail_log)} failures to: {LOG_FILE}")
