import os
import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime
from pathlib import Path

# TYPE = "predicate"  # Change to: "weighted", "anchored", etc.
# TYPE = "weighted"
TYPE = "anchored"

# === Settings ===
GRAPH_DIR = f"filtered_srl_graphs_{TYPE}"
LABEL_FILE = "subsentence_subcode_labels.json"
RESULT_DIR = f"results_srl_gcn_{TYPE}"
MAX_EPOCHS = 200
BATCH_SIZE = 16
LR = 1e-4
os.makedirs(RESULT_DIR, exist_ok=True)

# === Random Seed Logging ===
seed = random.randint(1, 100000)
random.seed(seed)
torch.manual_seed(seed)
print(f"🧪 Using random seed: {seed}")

# === Load Labels ===
with open(LABEL_FILE, "r", encoding="utf-8") as f:
    label_data = json.load(f)
labels_map = {
    k: v["label"] for k, v in label_data.items() if v["level"] == "subcode"
}
all_labels = sorted(set(labels_map.values()))
label2id = {l: i for i, l in enumerate(all_labels)}

# === Tokenizer ===
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# === Graph Dataset ===
class SRLDataset(Dataset):
    def __init__(self, graph_dir):
        self.graph_dir = Path(graph_dir)
        self.graph_files = sorted(list(self.graph_dir.glob("*.json")))

    def len(self):
        return len(self.graph_files)

    def get(self, idx):
        file = self.graph_files[idx]
        sid = file.stem
        with open(file, "r", encoding="utf-8") as f:
            graph = json.load(f)

        label_str = labels_map.get(sid)
        if label_str is None or label_str not in label2id:
            raise ValueError(f"Missing label for {sid}")
        y = torch.tensor(label2id[label_str], dtype=torch.long)

        tokens = [node["label"] for node in graph["nodes"] if "label" in node]
        inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt",
                           padding="max_length", truncation=True, max_length=64)
        x = inputs["input_ids"].squeeze(0)

        edge_index = torch.tensor(graph.get("edges", [])).t().contiguous() if graph.get("edges") else torch.empty((2, 0), dtype=torch.long)

        return Data(x=x, edge_index=edge_index, y=y)

dataset = SRLDataset(GRAPH_DIR)
total = len(dataset)
split = int(0.8 * total)
train_dataset = dataset[:split]
val_dataset = dataset[split:]

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# === Model ===
class SRLGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        with torch.no_grad():
            emb = self.bert(x)["last_hidden_state"].mean(dim=1)
        x = F.relu(self.conv1(emb, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.fc(x)

model = SRLGCN(in_dim=768, hidden_dim=128, num_classes=len(label2id))
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# === Training Loop ===
best_f1 = 0
train_log = []
errors = []

print("🚀 Training SRL-GCN model...")
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    losses = []
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    model.eval()
    preds, trues, texts = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            preds.extend(pred.tolist())
            trues.extend(batch.y.tolist())

    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average="macro")
    epoch_loss = sum(losses) / len(losses)
    print(f"📉 Epoch {epoch:02d} | Loss: {epoch_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")
    train_log.append({"epoch": epoch, "loss": epoch_loss, "accuracy": acc, "f1": f1})

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), os.path.join(RESULT_DIR, "best_model.pt"))

# === Save Outputs ===
with open(os.path.join(RESULT_DIR, "train_log.json"), "w") as f:
    json.dump({
        "model": "SRL-GCN",
        "label_type": "subcode",
        "input_file": LABEL_FILE,
        "graph_dir": GRAPH_DIR,
        "classes": all_labels,
        "seed": seed,
        "log": train_log
    }, f, indent=2)

# === Plot
epochs = [r["epoch"] for r in train_log]
accs = [r["accuracy"] for r in train_log]
f1s = [r["f1"] for r in train_log]

plt.figure(figsize=(10, 6))
plt.plot(epochs, accs, label="Accuracy")
plt.plot(epochs, f1s, label="Macro-F1")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("SRL-GCN Performance")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "training_plot.png"))
plt.close()

print("✅ Done. Results saved to:", RESULT_DIR)
