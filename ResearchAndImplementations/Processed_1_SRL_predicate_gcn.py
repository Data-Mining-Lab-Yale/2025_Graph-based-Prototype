# Processed_1_SRL_predicate_gcn.py (patched version with singleton class filtering)

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import random, numpy as np, os, json
import matplotlib.pyplot as plt
from collections import Counter
import torch
from torch_geometric.data import Data
torch.serialization.add_safe_globals([Data])


# ====== CONFIG ======
INPUT_FILE = "processed_graph_features_predicate.pt"
OUTPUT_DIR = "results_srl_predicate"
BATCH_SIZE = 32
NUM_EPOCHS = 200
LEARNING_RATE = 1e-3
USE_EDGE_WEIGHT = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====== Random seed ======
SEED = random.randint(1, 100000)
print(f"🧪 Using random seed: {SEED}")
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# ====== Load preprocessed data ======
print("📦 Loading processed graphs...")
# dataset = torch.load(INPUT_FILE)
dataset = torch.load(INPUT_FILE, weights_only=False)
all_labels = sorted(set([d.y for d in dataset]))
label2id = {label: i for i, label in enumerate(all_labels)}
for d in dataset:
    d.y = torch.tensor(label2id[d.y], dtype=torch.long)

# Count labels
strat_labels = [d.y.item() for d in dataset]
label_counts = Counter(strat_labels)
print(f"📦 Loaded graph dataset with {len(dataset)} samples and {len(label2id)} classes.")
print("📊 Class distribution (before filtering):", label_counts.most_common())

# ====== Filter out classes with <2 instances ======
valid_indices = [i for i in range(len(dataset)) if label_counts[dataset[i].y.item()] >= 2]
filtered_dataset = [dataset[i] for i in valid_indices]
filtered_labels = [dataset[i].y.item() for i in valid_indices]
print(f"🧹 Filtered dataset now has {len(filtered_dataset)} samples.")

# ====== Train-val split ======
train_idx, test_idx = train_test_split(
    list(range(len(filtered_dataset))),
    test_size=0.2,
    stratify=filtered_labels,
    random_state=SEED
)
train_loader = DataLoader([filtered_dataset[i] for i in train_idx], batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader([filtered_dataset[i] for i in test_idx], batch_size=BATCH_SIZE)

# ====== GCN Model ======
class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_weight, batch):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = global_mean_pool(x, batch)
        return self.lin(x)

model = GCN(in_dim=768, hidden_dim=128, out_dim=len(label2id))
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ====== Training ======
train_log = []
best_f1 = -1

print("🚀 Training SRL-GCN model...")
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr.view(-1), batch.batch)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Evaluation
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch.x, batch.edge_index, batch.edge_attr.view(-1), batch.batch)
            pred = out.argmax(dim=1)
            all_pred.extend(pred.tolist())
            all_true.extend(batch.y.tolist())

    acc = sum(p == t for p, t in zip(all_pred, all_true)) / len(all_true)
    f1 = f1_score(all_true, all_pred, average="macro")
    log = {
        "epoch": epoch,
        "loss": total_loss / len(train_loader),
        "accuracy": acc,
        "f1": f1
    }
    train_log.append(log)
    print(f"📉 Epoch {epoch:03d} | Loss: {log['loss']:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

    # Save best model
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))

# ====== Save logs and config ======
with open(os.path.join(OUTPUT_DIR, "train_log.json"), "w") as f:
    json.dump({
        "log": train_log,
        "seed": SEED,
        "label_type": "subcode",
        "input_file": INPUT_FILE,
        "label2id": label2id
    }, f, indent=2)

# ====== Save prediction errors ======
error_list = []
for pred, true, sample in zip(all_pred, all_true, [filtered_dataset[i] for i in test_idx]):
    if pred != true:
        error_list.append({
            "id": sample.id,
            "text": str(sample.x.shape),  # Placeholder for now
            "true_label": all_labels[true],
            "predicted_label": all_labels[pred]
        })
with open(os.path.join(OUTPUT_DIR, "errors.json"), "w") as f:
    json.dump(error_list, f, indent=2)

# ====== Plot training metrics ======
epochs = [l["epoch"] for l in train_log]
accs = [l["accuracy"] for l in train_log]
f1s = [l["f1"] for l in train_log]
losses = [l["loss"] for l in train_log]

plt.figure(figsize=(10, 6))
plt.plot(epochs, accs, label="Accuracy")
plt.plot(epochs, f1s, label="Macro-F1")
plt.plot(epochs, losses, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()
plt.title("SRL-GCN Predicate-Centric Training Curve")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_plot.png"))
plt.close()

print("✅ Done.")
