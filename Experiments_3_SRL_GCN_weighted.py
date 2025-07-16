import os
import json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertTokenizer, BertModel
import random
import numpy as np

# === Random seed ===
SEED = random.randint(1, 99999)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print(f"🧪 Using random seed: {SEED}")

# === CONFIG ===
GRAPH_DIR = "filtered_srl_graphs_weighted"
OUTPUT_DIR = "results_srl_gcn_weighted"
NUM_EPOCHS = 150
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
MAX_LEN = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# === Dataset ===
class SRLGraphDataset(Dataset):
    def __init__(self, graph_dir, tokenizer, label_encoder):
        self.graph_dir = graph_dir
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.graph_files = sorted([f for f in os.listdir(graph_dir) if f.endswith(".json")])

    def __len__(self):
        return len(self.graph_files)

    def __getitem__(self, idx):
        path = os.path.join(self.graph_dir, self.graph_files[idx])
        with open(path, "r", encoding="utf-8") as f:
            graph = json.load(f)

        label_str = graph["label"]
        text = graph.get("text", "")
        edges_raw = graph.get("edges", [])
        nodes = graph.get("nodes", [])

        # Tokenize the sentence text to generate one embedding vector
        encoded = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
        with torch.no_grad():
            outputs = bert_model(**{k: v.to(DEVICE) for k, v in encoded.items()})
            cls_vector = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        num_nodes = len(nodes)
        x = cls_vector.repeat(num_nodes, 1).squeeze(0)

        # Convert edge index (source, target) from string ids to integer indices
        node_id_to_index = {n["id"]: i for i, n in enumerate(nodes)}
        edge_index = []
        for e in edges_raw:
            src = node_id_to_index.get(e["source"])
            tgt = node_id_to_index.get(e["target"])
            if src is not None and tgt is not None:
                edge_index.append([src, tgt])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.zeros((2, 0), dtype=torch.long)

        y = torch.tensor(self.label_encoder.transform([label_str])[0], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y)
        data.text = text
        data.label_name = label_str
        return data

# === GCN Model ===
class GCNClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.linear(x)

# === Setup ===
print("🔧 Loading BERT and label encoder...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)
bert_model.eval()

labels = []
for f in os.listdir(GRAPH_DIR):
    with open(os.path.join(GRAPH_DIR, f), "r", encoding="utf-8") as g:
        labels.append(json.load(g)["label"])
label_encoder = LabelEncoder()
label_encoder.fit(labels)
num_classes = len(label_encoder.classes_)

# === Load Dataset ===
print("📦 Loading graph dataset...")
dataset = SRLGraphDataset(GRAPH_DIR, tokenizer, label_encoder)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader([train_dataset[i] for i in range(len(train_dataset))], batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader([test_dataset[i] for i in range(len(test_dataset))], batch_size=BATCH_SIZE)

# === Train ===
print("🚀 Training SRL-GCN model...")
model = GCNClassifier(in_channels=768, hidden_channels=128, num_classes=num_classes).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
train_log = []

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    model.eval()
    y_true, y_pred = [], []
    for batch in test_loader:
        batch = batch.to(DEVICE)
        with torch.no_grad():
            logits = model(batch.x, batch.edge_index, batch.batch)
            preds = logits.argmax(dim=1)
            y_true.extend(batch.y.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    train_log.append({"epoch": epoch + 1, "loss": avg_loss, "accuracy": acc, "f1": f1})
    print(f"📉 Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

# === Save Outputs ===
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))
with open(os.path.join(OUTPUT_DIR, "train_log.json"), "w", encoding="utf-8") as f:
    json.dump({
        "seed": SEED,
        "input_file": GRAPH_DIR,
        "model": "SRL-GCN-Weighted",
        "source": "subsentence",
        "label_type": "subcode",
        "class_names": list(label_encoder.classes_),
        "metrics": train_log
    }, f, indent=2)
print(f"✅ Training log saved to: {OUTPUT_DIR}/train_log.json")

# === Save Error Log ===
error_log = []
for batch in test_loader:
    batch = batch.to(DEVICE)
    with torch.no_grad():
        logits = model(batch.x, batch.edge_index, batch.batch)
        preds = logits.argmax(dim=1)
        for j in range(batch.y.size(0)):
            true = batch.y[j].item()
            pred = preds[j].item()
            error_log.append({
                "text": batch.text[j],
                "true_label": label_encoder.inverse_transform([true])[0],
                "pred_label": label_encoder.inverse_transform([pred])[0],
                "correct": pred == true
            })
with open(os.path.join(OUTPUT_DIR, "errors.json"), "w", encoding="utf-8") as f:
    json.dump({"seed": SEED, "errors": error_log}, f, indent=2)
print(f"📝 Prediction errors saved to: {OUTPUT_DIR}/errors.json")

# === Plot Training ===
epochs = [e["epoch"] for e in train_log]
losses = [e["loss"] for e in train_log]
accs = [e["accuracy"] for e in train_log]
f1s = [e["f1"] for e in train_log]
plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, label="Loss")
plt.plot(epochs, accs, label="Accuracy")
plt.plot(epochs, f1s, label="Macro-F1")
plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.title("SRL-GCN Weighted Training Progress")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_plot.png"))
plt.close()
print(f"📈 Plot saved to: {OUTPUT_DIR}/training_plot.png")

# Final log
print(f"\n✅ SRL-GCN-Weighted Final | Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")
