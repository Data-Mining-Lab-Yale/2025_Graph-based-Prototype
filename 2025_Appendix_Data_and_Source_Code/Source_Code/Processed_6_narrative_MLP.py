import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
import random
import numpy as np

# === Config ===
PT_FILE = "processed_graph_features_narrative_ego_mlp.pt"
OUTPUT_DIR = "results/results_narrative_ego_mlp"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TRAIN_LOG_FILE = os.path.join(OUTPUT_DIR, "narrative_ego_mlp_train_log.json")
ERROR_LOG_FILE = os.path.join(OUTPUT_DIR, "narrative_ego_mlp_errors.json")
MODEL_FILE = os.path.join(OUTPUT_DIR, "narrative_ego_mlp_msg_mlp_model.pt")
PLOT_FILE = os.path.join(OUTPUT_DIR, "narrative_ego_mlp_training_plot.png")

EPOCHS = 120
BATCH_SIZE = 32
LR = 1e-3
HIDDEN_DIM = 256
VAL_RATIO = 0.2
SEED = 91643

# === Reproducibility ===
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# === Label Mapping ===
def build_label_map(Y):
    labels = sorted(set(Y))
    return {label: i for i, label in enumerate(labels)}, {i: label for i, label in enumerate(labels)}

# === Load Data ===
def load_data(pt_file):
    data = torch.load(pt_file)
    X_raw, Y_raw, META = data["X"], data["Y"], data["meta"]
    label2id, id2label = build_label_map(Y_raw)
    X = torch.stack(X_raw)
    Y = torch.tensor([label2id[y] for y in Y_raw], dtype=torch.long)

    dataset = TensorDataset(X, Y)
    val_size = int(len(X) * VAL_RATIO)
    train_size = len(X) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    return train_set, val_set, label2id, id2label, META, Y_raw

# === MLP Classifier ===
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

# === Train + Evaluate ===
def train():
    train_set, val_set, label2id, id2label, meta_all, Y_raw = load_data(PT_FILE)
    model = MLPClassifier(768, HIDDEN_DIM, len(label2id))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    logs = []
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                out = model(xb)
                preds.extend(out.argmax(dim=1).tolist())
                targets.extend(yb.tolist())

        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average="macro")
        logs.append({
            "epoch": epoch + 1,
            "loss": round(total_loss, 6),
            "accuracy": round(acc, 6),
            "f1": round(f1, 6)
        })
        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

    # Save model
    torch.save(model.state_dict(), MODEL_FILE)

    # Save training plot
    plt.figure()
    plt.plot([log["epoch"] for log in logs], [log["loss"] for log in logs], label="Loss")
    plt.plot([log["epoch"] for log in logs], [log["accuracy"] for log in logs], label="Accuracy")
    plt.plot([log["epoch"] for log in logs], [log["f1"] for log in logs], label="Macro F1")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Training Curve (MSG-MLP)")
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_FILE)
    plt.close()

    # Save training log JSON
    train_log = {
        "log": logs,
        "seed": SEED,
        "input_file": os.path.basename(PT_FILE),
        "label_type": "subcode",
        "label2id": label2id
    }
    with open(TRAIN_LOG_FILE, "w") as f:
        json.dump(train_log, f, indent=2)

    # Save error log
    val_indices = val_set.indices if hasattr(val_set, 'indices') else range(len(val_set))
    val_X = torch.stack([val_set[i][0] for i in range(len(val_set))])
    val_Y = torch.tensor([val_set[i][1] for i in range(len(val_set))])
    with torch.no_grad():
        logits = model(val_X)
        pred_ids = logits.argmax(dim=1).tolist()
        true_ids = val_Y.tolist()

    errors = []
    for i, pred, true in zip(val_indices, pred_ids, true_ids):
        if pred != true:
            errors.append({
                "graph_id": meta_all[i]["graph_id"],
                "center_id": meta_all[i]["center_id"],
                "original_text": meta_all[i]["original_text"],
                "true_label": id2label[true],
                "pred_label": id2label[pred]
            })
    with open(ERROR_LOG_FILE, "w") as f:
        json.dump(errors, f, indent=2)

if __name__ == "__main__":
    train()
