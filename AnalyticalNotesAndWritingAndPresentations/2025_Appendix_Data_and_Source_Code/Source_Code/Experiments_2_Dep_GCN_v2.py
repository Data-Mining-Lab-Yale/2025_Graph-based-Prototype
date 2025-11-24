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
from torch.utils.data import random_split
import random
import numpy as np

# === Generate and fix random seed ===
SEED = random.randint(1, 99999)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


print(f"🧪 Using random seed: {SEED}")
# === CONFIG ===
GRAPH_DIR = "filtered_dependency_graphs"
OUTPUT_DIR = "results_dep_gcn"
NUM_EPOCHS = 150
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
MAX_LEN = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# === Dataset Loader ===
class DependencyGraphDataset(Dataset):
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

        raw_nodes = graph.get("nodes", [])
        tokens = [n["label"] for n in raw_nodes if isinstance(n, dict) and "label" in n]
        label = graph["label"]
        text = graph.get("text", "")

        if not tokens:
            raise IndexError("No valid tokens")

        try:
            encoded = self.tokenizer(tokens, is_split_into_words=True, return_tensors="pt",
                                     padding=True, truncation=True, max_length=MAX_LEN)
            with torch.no_grad():
                outputs = bert_model(**{k: v.to(DEVICE) for k, v in encoded.items()})
                # ✅ Corrected: preserve full sequence for node features
                node_features = outputs.last_hidden_state.squeeze(0)
        except Exception as e:
            raise IndexError(f"Tokenization failed: {e}")

        edges = graph.get("edges", [])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2, 0), dtype=torch.long)

        y = torch.tensor(self.label_encoder.transform([label])[0], dtype=torch.long)
        data = Data(x=node_features, edge_index=edge_index, y=y)
        data.text = text
        data.label_name = label
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
print("🔧 Loading BERT and labels...")
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

# === Load dataset ===
print("📦 Loading graph dataset...")
dataset = DependencyGraphDataset(GRAPH_DIR, tokenizer, label_encoder)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader([train_dataset[i] for i in range(len(train_dataset))],
                          batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader([test_dataset[i] for i in range(len(test_dataset))],
                         batch_size=BATCH_SIZE)

# === Train ===
print("🚀 Training DEP-GCN model...")
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
# train_summary = {
#     "metrics": train_log,
#     "class_names": list(label_encoder.classes_)
# }
# with open(os.path.join(OUTPUT_DIR, "train_log.json"), "w", encoding="utf-8") as f:
#     json.dump(train_summary, f, indent=2)
train_summary = {
    "seed": SEED,
    "metrics": train_log,
    "class_names": list(label_encoder.classes_)
}
with open(os.path.join(OUTPUT_DIR, "train_log.json"), "w", encoding="utf-8") as f:
    json.dump(train_summary, f, indent=2)


# with open(os.path.join(OUTPUT_DIR, "train_log.json"), "w", encoding="utf-8") as f:
#     json.dump(train_log, f, indent=2)
print(f"✅ Training log saved to: {OUTPUT_DIR}/train_log.json")

# # === Plot Training Curve ===
# epochs = [e["epoch"] for e in train_log]
# losses = [e["loss"] for e in train_log]
# f1s = [e["f1"] for e in train_log]
# plt.figure()
# plt.plot(epochs, losses, label="Loss")
# plt.plot(epochs, f1s, label="Macro-F1")
# plt.xlabel("Epoch")
# plt.ylabel("Metric")
# plt.legend()
# plt.title("DEP-GCN Training")
# plt.savefig(os.path.join(OUTPUT_DIR, "training_plot.png"))
# print(f"📈 Plot saved to: {OUTPUT_DIR}/training_plot.png")

# === Plot Training Curve ===
epochs = [e["epoch"] for e in train_log]
losses = [e["loss"] for e in train_log]
f1s = [e["f1"] for e in train_log]
accs = [e["accuracy"] for e in train_log]

plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, label="Loss", color="tab:blue")
plt.plot(epochs, accs, label="Accuracy", color="tab:orange")
plt.plot(epochs, f1s, label="Macro-F1", color="tab:green")
plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.title("DEP-GCN Training Progress")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_plot.png"), dpi=300)
plt.close()


# === Save Prediction Errors ===
error_log = []
for batch in test_loader:
    batch = batch.to(DEVICE)
    with torch.no_grad():
        out = model(batch.x, batch.edge_index, batch.batch)
        preds = out.argmax(dim=1)
        for j in range(batch.y.size(0)):
            true = batch.y[j].item()
            pred = preds[j].item()
            error_log.append({
                "text": batch.text[j],
                "true_label": label_encoder.inverse_transform([true])[0],
                "pred_label": label_encoder.inverse_transform([pred])[0],
                "correct": pred == true
            })

# with open(os.path.join(OUTPUT_DIR, "errors.json"), "w", encoding="utf-8") as f:
#     json.dump(error_log, f, indent=2)
with open(os.path.join(OUTPUT_DIR, "errors.json"), "w", encoding="utf-8") as f:
    json.dump({
        "seed": SEED,
        "errors": error_log
    }, f, indent=2)

print(f"📝 Prediction errors saved to: {OUTPUT_DIR}/errors.json")

# === Final Results ===
print(f"\n✅ DEP-GCN Final Evaluation | Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")
