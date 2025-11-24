import os
import random
import json
import torch
import numpy as np
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.serialization import add_safe_globals
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Set random seed
SEED = random.randint(0, 100000)
print(f"🧪 Using random seed: {SEED}")
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ========== Load Data ==========
INPUT_FILE = "dep_graph_features.pt"
OUTPUT_DIR = Path("results_dep_gcn/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("📦 Loading processed graph data...")
# dataset = torch.load(INPUT_FILE)
add_safe_globals([Data, DataEdgeAttr])
dataset = torch.load(INPUT_FILE, weights_only=False)
# Filter out invalid labels
dataset = [d for d in dataset if hasattr(d, 'y') and d.y != -1]
print(f"✅ Loaded {len(dataset)} valid graphs")
# for d in dataset:
#     d.y = d.y_encoded

# ========== Split ==========
random.shuffle(dataset)
split1 = int(0.8 * len(dataset))
split2 = int(0.9 * len(dataset))
train_data = dataset[:split1]
val_data = dataset[split1:split2]
test_data = dataset[split2:]

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

# ========== GCN Model ==========
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.lin(x)

input_dim = dataset[0].x.shape[1]
output_dim = int(max([int(d.y) for d in dataset])) + 1
model = GCN(in_channels=input_dim, hidden_channels=64, out_channels=output_dim)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# ========== Train ==========
train_losses, val_losses, val_accs = [], [], []
best_val_acc = 0
best_model_state = None

for epoch in range(1, 21):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = loss_fn(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_losses.append(total_loss / len(train_loader))

    # Validation
    model.eval()
    correct = 0
    total = 0
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            out = model(batch.x, batch.edge_index, batch.batch)
            val_loss += loss_fn(out, batch.y).item()
            preds = out.argmax(dim=1)
            correct += (preds == batch.y).sum().item()
            total += batch.y.size(0)
    acc = correct / total
    val_losses.append(val_loss / len(val_loader))
    val_accs.append(acc)

    print(f"Epoch {epoch}: Train Loss={train_losses[-1]:.4f}, Val Loss={val_losses[-1]:.4f}, Val Acc={acc:.4f}")
    
    if acc > best_val_acc:
        best_val_acc = acc
        best_model_state = model.state_dict()

# Save best model
torch.save(best_model_state, OUTPUT_DIR / "best_model.pt")

# ========== Test ==========
model.load_state_dict(best_model_state)
model.eval()

error_log = []
correct_log = []

with torch.no_grad():
    for batch in test_loader:
        out = model(batch.x, batch.edge_index, batch.batch)
        preds = out.argmax(dim=1)
        for i in range(len(batch.y)):
            idx = batch.ptr.tolist().index(i) if i < len(batch.ptr.tolist()) else 0
            item = {
                "text": batch.text[i],
                "label": int(batch.y_raw[i]),
                "prediction": int(preds[i]),
                "correct": int(batch.y_raw[i]) == int(preds[i]),
                "message_id": batch.message_id[i],
                "subsentence_index": batch.subsentence_index[i],
                "seed": SEED
            }
            if item["correct"]:
                correct_log.append(item)
            else:
                error_log.append(item)

with open(OUTPUT_DIR / "dep_gcn_errors.json", "w", encoding="utf-8") as f:
    json.dump(error_log, f, ensure_ascii=False, indent=2)

with open(OUTPUT_DIR / "dep_gcn_correct.json", "w", encoding="utf-8") as f:
    json.dump(correct_log, f, ensure_ascii=False, indent=2)

# ========== Plot ==========
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.plot(val_accs, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("DEP GCN")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "dep_gcn_plot.png")
plt.close()
