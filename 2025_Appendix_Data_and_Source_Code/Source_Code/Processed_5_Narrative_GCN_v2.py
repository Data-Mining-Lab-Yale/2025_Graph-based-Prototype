import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
from sklearn.metrics import f1_score, accuracy_score
import random, os, json
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# ========== Config ==========
INPUT_FILE = "processed_graph_features_narrative_ego_amr.pt"
RESULT_DIR = Path("results/results_narrative_ego_amr")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
EPOCHS = 120
BATCH_SIZE = 16
SEED = random.randint(1, 99999)
torch.manual_seed(SEED)
print(f"🧪 Using random seed: {SEED}")

# ========== Load Data ==========
dataset = torch.load(INPUT_FILE, weights_only=False)
labels = list(sorted({data.y for data in dataset}))
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

for data in dataset:
    data.y = torch.tensor([label2id[data.y]])

# Split
random.shuffle(dataset)
n = len(dataset)
train_dataset = dataset[:int(0.8 * n)]
val_dataset = dataset[int(0.8 * n):int(0.9 * n)]
test_dataset = dataset[int(0.9 * n):]

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=1)

# ========== Model ==========
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

model = GCN(dataset[0].x.size(1), 64, len(label2id))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.NLLLoss()

# ========== Training ==========
best_f1 = 0
train_log = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    preds, gts = [], []

    for data in train_loader:
        optimizer.zero_grad()
        # out = model(data.x, data.edge_index, edge_weight=data.edge_attr.squeeze())
        # Validate edge weights
        # ✅ Safe edge weight extraction
        if hasattr(data, "edge_attr") and isinstance(data.edge_attr, torch.Tensor) \
        and data.edge_attr.ndim == 1 and data.edge_attr.numel() == data.edge_index.size(1):
            edge_weight = data.edge_attr
        else:
            edge_weight = None
        out = model(data.x, data.edge_index, edge_weight=edge_weight)
        
        # out = out[data.batch == 0]  # center node prediction
        center_node_indices = []
        num_graphs = data.num_graphs
        for i in range(num_graphs):
            idx = (data.batch == i).nonzero(as_tuple=True)[0][0]
            center_node_indices.append(idx.item())
        out = out[center_node_indices]
        
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.y.size(0)
        preds += pred.tolist()
        gts += data.y.view(-1).tolist()

    acc = correct / total
    f1 = f1_score(gts, preds, average="macro")
    print(f"Epoch {epoch:2d} | Loss: {total_loss:.3f} | Acc: {acc:.3f} | F1: {f1:.3f}")
    train_log.append({"epoch": epoch, "loss": total_loss, "accuracy": acc, "f1": f1})

    # Save best
    if f1 > best_f1:
        torch.save(model.state_dict(), RESULT_DIR / "best_model.pt")
        best_f1 = f1

# ========== Evaluation ==========
model.load_state_dict(torch.load(RESULT_DIR / "best_model.pt"))
model.eval()
preds, gts, texts, ids, trues = [], [], [], [], []

# for data in test_loader:
#     # out = model(data.x, data.edge_index, edge_weight=data.edge_attr.squeeze())
#     if hasattr(data, "edge_attr") and isinstance(data.edge_attr, torch.Tensor) \
#     and data.edge_attr.ndim == 1 and data.edge_attr.numel() == data.edge_index.size(1):
#         edge_weight = data.edge_attr
#     else:
#         edge_weight = None
#     out = model(data.x, data.edge_index, edge_weight=edge_weight)    

#     pred = out.argmax(dim=1)
#     true = data.y.view(-1)
#     preds.append(pred.item())
#     # preds.append(pred.argmax().item())
#     gts.append(true.item())
#     ids.append(data.id[0] if isinstance(data.id, list) else data.id)
#     texts.append(data.text[0] if isinstance(data.text, list) else data.text)
#     trues.append(true.item())

for data in test_loader:
    if hasattr(data, "edge_attr") and isinstance(data.edge_attr, torch.Tensor) \
    and data.edge_attr.ndim == 1 and data.edge_attr.numel() == data.edge_index.size(1):
        edge_weight = data.edge_attr
    else:
        edge_weight = None

    out = model(data.x, data.edge_index, edge_weight=edge_weight)

    # ✅ Extract center node prediction
    idx = (data.batch == 0).nonzero(as_tuple=True)[0][0]
    pred = out[idx].argmax(dim=0).item()
    true = data.y.item()

    preds.append(pred)
    gts.append(true)
    ids.append(data.id[0] if isinstance(data.id, list) else data.id)
    texts.append(data.text[0] if isinstance(data.text, list) else data.text)
    trues.append(true)


acc = accuracy_score(gts, preds)
f1 = f1_score(gts, preds, average="macro")
print(f"\n🎯 Test Accuracy: {acc:.3f} | Test F1: {f1:.3f}")

# ========== Save Logs ==========
with open(RESULT_DIR / "train_log.json", "w") as f:
    json.dump({
        "log": train_log,
        "seed": SEED,
        "input_file": INPUT_FILE,
        "label_type": "subcode",
        "label2id": label2id
    }, f, indent=2)

# ========== Save Errors ==========
errors = []
for i in range(len(preds)):
    errors.append({
        "id": ids[i],
        "text": texts[i],
        "true_label": id2label[gts[i]],
        "predicted_label": id2label[preds[i]],
        "correct": gts[i] == preds[i]
    })
with open(RESULT_DIR / "errors.json", "w") as f:
    json.dump(errors, f, indent=2)

# ========== Plot ==========
epochs = [entry["epoch"] for entry in train_log]
losses = [entry["loss"] for entry in train_log]
accs = [entry["accuracy"] for entry in train_log]
f1s = [entry["f1"] for entry in train_log]

# plt.figure(figsize=(8, 5))
# plt.plot(epochs, accs, label="Accuracy")
# plt.plot(epochs, f1s, label="Macro-F1")
# plt.plot(epochs, losses, label="Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Score")
# plt.title("Narrative GCN Training")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(RESULT_DIR / "training_plot.png")
# plt.close()


epochs = [entry["epoch"] for entry in train_log]
losses = [entry["loss"] for entry in train_log]
accs = [entry["accuracy"] for entry in train_log]
f1s = [entry["f1"] for entry in train_log]

fig, ax1 = plt.subplots(figsize=(8, 5))

# Accuracy and F1
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy / F1", color="tab:orange")
# ax1.plot(epochs, accs, label="Accuracy", color="tab:blue", linestyle='-', marker='o')
# ax1.plot(epochs, f1s, label="Macro-F1", color="tab:orange", linestyle='--', marker='x')
ax1.plot(epochs, accs, label="Accuracy", color="tab:orange")
ax1.plot(epochs, f1s, label="Macro-F1", color="tab:green")
ax1.tick_params(axis='y', labelcolor="tab:blue")
ax1.set_ylim(0.0, 1.0)

# Loss on secondary axis
ax2 = ax1.twinx()
ax2.set_ylabel("Loss", color="tab:blue")
# ax2.plot(epochs, losses, label="Loss", color="tab:red", linestyle='-.')
ax2.plot(epochs, losses, label="Loss", color="tab:blue")
ax2.tick_params(axis='y', labelcolor="tab:blue")

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

# Title and layout
plt.title("Narrative GCN Training Performance")
plt.grid(True)
plt.tight_layout()

# Save
plt.savefig(RESULT_DIR / "training_plot.png")
plt.close()