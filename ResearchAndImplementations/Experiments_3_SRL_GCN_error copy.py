import os
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# ====== Configuration ======
TYPE = "anchored"  # Choose from: "anchored", "predicate", "weighted"
GRAPH_DIR = f"filtered_srl_graphs_{TYPE}"
RESULT_DIR = f"results_srl_gcn_{TYPE}"
LABEL_TYPE = "subcode"
BATCH_SIZE = 32
EPOCHS = 150
PATIENCE = 10
SEED = random.randint(1, 99999)

os.makedirs(RESULT_DIR, exist_ok=True)

# ====== Set seed ======
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

set_seed(SEED)
print(f"🧪 Using random seed: {SEED}")

# ====== Dataset loader ======
class SRLDataset(Dataset):
    def __init__(self, graph_dir):
        super().__init__()
        self.graph_dir = graph_dir
        self.graph_files = sorted([f for f in os.listdir(graph_dir) if f.endswith(".json")])

    def len(self):
        return len(self.graph_files)

    def get(self, idx):
        fname = self.graph_files[idx]
        fpath = os.path.join(self.graph_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            graph = json.load(f)

        node_ids = []
        node_features = []
        node_map = {}
        for i, node in enumerate(graph["nodes"]):
            node_id = node["id"]
            node_ids.append(node_id)
            node_map[node_id] = i
            node_features.append([1.0] * 16)  # fixed dummy vector

        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = [[], []]
        for edge in graph["edges"]:
            src = node_map.get(edge["source"])
            tgt = node_map.get(edge["target"])
            if src is not None and tgt is not None:
                edge_index[0].append(src)
                edge_index[1].append(tgt)
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        label_str = graph["label"]
        return Data(x=x, edge_index=edge_index, y=label_str, id=fname.replace(".json", ""))

# ====== Load dataset ======
dataset = SRLDataset(GRAPH_DIR)
all_label_names = sorted(set([data.y for data in dataset]))
label2id = {label: i for i, label in enumerate(all_label_names)}
id2label = {i: label for label, i in label2id.items()}

# Convert string labels to int
for i in range(len(dataset)):
    dataset[i].y = torch.tensor(label2id[dataset[i].y], dtype=torch.long)

print(f"📦 Loaded graph dataset with {len(dataset)} samples and {len(label2id)} classes.")

# ====== Stratified split ======
indices = list(range(len(dataset)))
strat_labels = [dataset[i].y.item() for i in indices]
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=SEED, stratify=strat_labels)
train_dataset = [dataset[i] for i in train_idx]
test_dataset = [dataset[i] for i in test_idx]
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ====== Model ======
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)

# ====== Training ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCN(in_channels=16, hidden_channels=64, out_channels=len(label2id)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

train_losses, train_accs, train_f1s = [], [], []
best_f1 = 0
no_improve = 0
best_model_path = os.path.join(RESULT_DIR, "best_model.pt")

print("🚀 Training SRL-GCN model...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_true = []
    all_pred = []

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        preds = out.argmax(dim=1)
        correct += (preds == batch.y).sum().item()
        total += batch.num_graphs
        all_true.extend(batch.y.cpu().tolist())
        all_pred.extend(preds.cpu().tolist())

    acc = correct / total
    report = classification_report(all_true, all_pred, target_names=[id2label[i] for i in range(len(label2id))], output_dict=True, zero_division=0)
    f1 = report["macro avg"]["f1-score"]

    train_losses.append(total_loss / total)
    train_accs.append(acc)
    train_f1s.append(f1)

    print(f"📉 Epoch {epoch:03d} | Loss: {total_loss / total:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        no_improve = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"💾 New best model saved to: {best_model_path}")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"🛑 Early stopping at epoch {epoch}. No improvement for {PATIENCE} epochs.")
            break

# ====== Evaluation ======
model.load_state_dict(torch.load(best_model_path))
model.eval()
all_true, all_pred, all_ids = [], [], []
errors = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        preds = out.argmax(dim=1)

        all_true.extend(batch.y.cpu().tolist())
        all_pred.extend(preds.cpu().tolist())
        all_ids.extend(batch.id)

        for i in range(len(preds)):
            if preds[i].item() != batch.y[i].item():
                errors.append({
                    "id": batch.id[i],
                    "true_label": id2label[batch.y[i].item()],
                    "predicted_label": id2label[preds[i].item()]
                })

accuracy = sum([p == t for p, t in zip(all_pred, all_true)]) / len(all_true)
macro_f1 = classification_report(all_true, all_pred, output_dict=True, zero_division=0)["macro avg"]["f1-score"]

train_log = {
    "label_type": LABEL_TYPE,
    "input_file": GRAPH_DIR,
    "seed": SEED,
    "class_names": [id2label[i] for i in range(len(label2id))],
    "train_loss": train_losses,
    "train_acc": train_accs,
    "train_f1": train_f1s,
    "final_accuracy": accuracy,
    "final_f1": macro_f1
}

with open(os.path.join(RESULT_DIR, "train_log.json"), "w") as f:
    json.dump(train_log, f, indent=2)

with open(os.path.join(RESULT_DIR, "errors.json"), "w") as f:
    json.dump(errors, f, indent=2)

print(f"📊 Metrics saved to: {os.path.join(RESULT_DIR, 'train_log.json')}")
print(f"📝 Prediction errors saved to: {os.path.join(RESULT_DIR, 'errors.json')}")
print(f"\n✅ SRL-GCN Final Evaluation | Accuracy: {accuracy:.4f} | F1 Score: {macro_f1:.4f}")

# ====== Plotting ======
plt.figure()
plt.plot(train_losses, label="Loss")
plt.plot(train_accs, label="Accuracy")
plt.plot(train_f1s, label="Macro F1")
plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.title("SRL-GCN Training Performance")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "training_plot.png"))
print(f"📈 Plot saved to: {os.path.join(RESULT_DIR, 'training_plot.png')}")
