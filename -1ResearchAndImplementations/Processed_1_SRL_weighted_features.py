import os
import json
import torch
from tqdm import tqdm
from torch_geometric.data import Data
from transformers import BertTokenizer, BertModel

# Configurable flags
USE_BERT = True
USE_EDGE_WEIGHT = True

# Paths
GRAPH_DIR = "filtered_srl_graphs_weighted"  # You can replace this with any graph folder
OUTPUT_FILE = "processed_graph_features.pt"

# Load BERT model (only if USE_BERT)
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
        return torch.zeros(768)  # Placeholder if not using BERT

# Process a single graph JSON file
def process_graph_file(path):
    with open(path, "r", encoding="utf-8") as f:
        graph = json.load(f)

    node_id_map = {}
    x_list = []
    for i, node in enumerate(graph["nodes"]):
        node_id = node["id"]
        node_id_map[node_id] = i
        if "label" in node:
            x_list.append(encode_text(node["label"]))
        else:
            x_list.append(torch.zeros(768))  # fallback
    x = torch.stack(x_list)

    # Process edges and optional edge weights
    edge_index = []
    edge_attr = []
    for edge in graph.get("edges", []):
        src = node_id_map[edge["source"]]
        tgt = node_id_map[edge["target"]]
        edge_index.append([src, tgt])
        if USE_EDGE_WEIGHT and "weight" in edge:
            edge_attr.append([float(edge["weight"])])
        else:
            edge_attr.append([1.0])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    label = graph["label"]
    sid = graph.get("sentence_id", os.path.basename(path).replace(".json", ""))
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=label, id=sid)

# Batch process
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
    print(f"✅ Saved {len(dataset)} processed graphs to {output_file}")

# Run
if __name__ == "__main__":
    process_all_graphs(GRAPH_DIR, OUTPUT_FILE)
