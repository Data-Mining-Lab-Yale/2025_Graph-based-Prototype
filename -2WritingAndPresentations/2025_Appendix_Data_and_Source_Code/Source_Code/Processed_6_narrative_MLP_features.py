import os
import json
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

# === Config ===
GRAPH_DIR = "filtered_narrative_ego_graphs"             # 📁 set this to your narrative graph folder
OUTPUT_FILE = "processed_graph_features_narrative_ego_mlp.pt"       # 💾 output file
USE_BERT = True

# === BERT Setup ===
if USE_BERT:
    print("🔤 Loading BERT model...")
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
        return torch.zeros(768)  # dummy feature

# === Process one graph file ===
def process_graph_file(path):
    with open(path, "r", encoding="utf-8") as f:
        graph = json.load(f)

    center_id = graph["center_id"]
    label = graph["label"]
    graph_id = os.path.basename(path).replace(".json", "")
    text = graph.get("text", "")

    # Encode all node texts
    node_feat_map = {
        node["id"]: encode_text(node.get("label", ""))
        for node in graph.get("nodes", [])
    }

    if center_id not in node_feat_map:
        raise ValueError(f"Center ID '{center_id}' not found in nodes for {graph_id}")

    # Weighted pooling over neighbors + self
    pooled = torch.zeros_like(node_feat_map[center_id])
    total_weight = 0.0

    for edge in graph.get("edges", []):
        src, tgt = edge["source"], edge["target"]
        weight = float(edge.get("weight", 1.0))

        if src == center_id and tgt in node_feat_map:
            pooled += weight * node_feat_map[tgt]
            total_weight += weight

    # fallback: use center only
    pooled_feat = (
        pooled / total_weight if total_weight > 0 else node_feat_map[center_id]
    )

    meta = {
        "graph_id": graph_id,
        "center_id": center_id,
        "original_text": text,
        "label": label,
    }

    return pooled_feat, label, meta

# === Batch process all graphs ===
def process_all_graphs(graph_dir, output_file):
    X, Y, META = [], [], []
    files = sorted([f for f in os.listdir(graph_dir) if f.endswith(".json")])
    print(f"📂 Found {len(files)} graph files in '{graph_dir}'")

    for fname in tqdm(files, desc="📊 Processing MSG-MLP features"):
        fpath = os.path.join(graph_dir, fname)
        try:
            feat, label, meta = process_graph_file(fpath)
            X.append(feat)
            Y.append(label)
            META.append(meta)
        except Exception as e:
            print(f"⚠️ Error in {fname}: {e}")

    torch.save({"X": X, "Y": Y, "meta": META}, output_file)
    print(f"✅ Saved {len(X)} pooled features to {output_file}")

# === Run ===
if __name__ == "__main__":
    process_all_graphs(GRAPH_DIR, OUTPUT_FILE)
