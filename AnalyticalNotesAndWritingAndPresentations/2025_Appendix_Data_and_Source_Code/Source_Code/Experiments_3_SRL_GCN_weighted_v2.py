import os
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from datetime import datetime

# === Settings ===
GRAPH_DIR = "filtered_srl_graphs_weighted"
RESULT_DIR = "results_srl_weighted"
os.makedirs(RESULT_DIR, exist_ok=True)
LEARNING_RATE = 1e-4
EPOCHS = 3000
BATCH_SIZE = 32

# === Random Seed ===
SEED = random.randint(1, 100000)
torch.manual_seed(SEED)
random.seed(SEED)
print(f"🧪 Using random seed: {SEED}")

# === Dataset ===
class SRLDataset(Dataset):
    def __init__(self, graph_dir):
        super().__init__()
        self.graph_dir = graph_dir
        self.graph_files = [f for f in os.listdir(graph_dir) if f.endswith(".json")]
        self.label2id = {}
        self.id2label = {}

        for fname in self.graph_files:
            with open(os.path.join(graph_dir, fname), "r") as f:
                label = json.load(f)["label"]
                if label not in self.label2id:
                    idx = len(self.label2id)
                    self.label2id[label] = idx
                    self.id2label[idx] = label

    def len(self):
        return len(self.graph_files)

    def get(self, idx):
        fname = self.graph_files[idx]
        with open(os.path.join(self.graph_dir, fname), "r") as f:
            graph = json.load(f)

        node_id_map = {node["id"]: i for i, node in enumerate(graph["nodes"])}
        x = torch.eye(len(graph["nodes"]))  # identity feature
        edge_index = []
        edge_weight = []

        for edge in graph.get("edges", []):
            src = node_id_map[edge["source"]]
            tgt = node_id_map[edge["target"]]
            edge_index.append([src, tgt])
            edge_weight.append(edge.get("weight", 1.0))

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)
        y = torch.tensor(self.label2id[graph["label"]], dtype=torch.long)
        return Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y)

# === GCN Model ===
class SRLGCNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_weight, batch):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = global_mean_pool(x, batch)
        return self.lin(x)

# === Load dataset ===
dataset = SRLDataset(GRAPH_DIR)
indices = list(range(len(dataset)))
split = int(0.8 * len(indices))
train_idx, test_idx = indices[:split], indices[split:]

train_dataset = [dataset.get(i) for i in train_idx]
test_dataset = [dataset.get(i) for i in test_idx]
print(f"📦 Loaded graph dataset with {len(dataset)} samples and {len(dataset.label2id)} classes.")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# === Training ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SRLGCNModel(in_channels=dataset.get(0).x.size(1), hidden_channels=64, num_classes=len(dataset.label2id)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_log = []
errors = []
best_f1 = -1

print("🚀 Training SRL-GCN model...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
            preds = out.argmax(dim=1)
            all_pred.extend(preds.tolist())
            all_true.extend(batch.y.tolist())
            for i in range(len(batch.y)):
                if preds[i] != batch.y[i]:
                    errors.append({
                        "true_label": dataset.id2label[batch.y[i].item()],
                        "predicted_label": dataset.id2label[preds[i].item()]
                    })

    acc = sum([p == t for p, t in zip(all_pred, all_true)]) / len(all_true)
    f1 = f1_score(all_true, all_pred, average="macro")
    train_log.append({"epoch": epoch, "loss": total_loss / len(train_loader), "accuracy": acc, "f1": f1})
    print(f"📉 Epoch {epoch:03d} | Loss: {total_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), os.path.join(RESULT_DIR, "best_model.pt"))

# === Save Outputs ===
with open(os.path.join(RESULT_DIR, "train_log.json"), "w") as f:
    json.dump({
        "seed": SEED,
        "learning_rate": LEARNING_RATE,
        "timestamp": datetime.now().isoformat(),
        "label_type": "subcode",
        "label_names": dataset.id2label,
        "log": train_log
    }, f, indent=2)

with open(os.path.join(RESULT_DIR, "errors.json"), "w") as f:
    json.dump(errors, f, indent=2)

# === Plotting ===
plt.figure()
plt.plot([x["accuracy"] for x in train_log], label="Accuracy")
plt.plot([x["f1"] for x in train_log], label="F1 Score")
plt.plot([x["loss"] for x in train_log], label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.legend()
plt.title("SRL-GCN Weighted Graph Training")
plt.savefig(os.path.join(RESULT_DIR, "training_plot.png"))
plt.close()

print(f"✅ Final Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")
