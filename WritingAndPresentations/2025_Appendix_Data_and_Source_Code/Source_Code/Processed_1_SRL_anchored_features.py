# Processed_3_SRL_anchored.py
import os
import json
import torch
from tqdm import tqdm
from torch_geometric.data import Data
from transformers import BertTokenizer, BertModel

# === Config ===
USE_BERT = True
GRAPH_DIR = "filtered_srl_graphs_anchored"  # folder with .json input
OUTPUT_FILE = "processed_graph_features_anchored.pt"

# === BERT Setup ===
if USE_BERT:
    print("🔤 Loading BERT...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_model.eval()

    def encode_text(text):
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=32)
            outputs = bert_model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).squeeze(0)
else:
    def encode_text(text):
        return torch.zeros(768)

# === Process Single Graph ===
def process_graph_file(path):
    with open(path, "r", encoding="utf-8") as f:
        graph = json.load(f)

    node_id_map = {}
    x_list = []
    for i, node in enumerate(graph["nodes"]):
        node_id = node["id"]
        node_id_map[node_id] = i
        x_list.append(encode_text(node.get("label", "")))
    x = torch.stack(x_list)

    edge_index = []
    for edge in graph.get("edges", []):
        src = node_id_map.get(edge["source"])
        tgt = node_id_map.get(edge["target"])
        if src is not None and tgt is not None:
            edge_index.append([src, tgt])
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    edge_attr = torch.ones((edge_index.size(1), 1))  # all edge weights = 1.0

    label = graph.get("label", "Unknown")
    sid = graph.get("sentence_id", os.path.basename(path).replace(".json", ""))
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=label, id=sid)

# === Batch Processing ===
def process_all_graphs(graph_dir, output_file):
    dataset = []
    files = [f for f in os.listdir(graph_dir) if f.endswith(".json")]
    print(f"📂 Found {len(files)} graph files.")
    for fname in tqdm(files, desc="📊 Processing graphs"):
        try:
            data = process_graph_file(os.path.join(graph_dir, fname))
            dataset.append(data)
        except Exception as e:
            print(f"⚠️ Error in {fname}: {e}")
    torch.save(dataset, output_file)
    print(f"✅ Saved {len(dataset)} graphs to {output_file}")

# === Run ===
if __name__ == "__main__":
    process_all_graphs(GRAPH_DIR, OUTPUT_FILE)
