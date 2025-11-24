import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import random

# === Config ===
PT_FILE = "processed_graph_features_narrative_ego_mlp.pt"
OUTPUT_DIR = "results/results_narrative_ego_mlp_2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_FILE = os.path.join(OUTPUT_DIR, "narrative_ego_mlp_msg_mlp_model.pt")
PLOT_FILE = os.path.join(OUTPUT_DIR, "narrative_ego_mlp_training_plot.png")
TRAIN_LOG_FILE = os.path.join(OUTPUT_DIR, "narrative_ego_mlp_train_log.json")
TRAIN_ERR_FILE = os.path.join(OUTPUT_DIR, "narrative_ego_mlp_errors.json")
TEST_LOG_FILE = os.path.join(OUTPUT_DIR, "narrative_ego_mlp_test_log.json")
TEST_ERR_FILE = os.path.join(OUTPUT_DIR, "narrative_ego_mlp_test_errors.json")

EPOCHS = 120
BATCH_SIZE = 32
LR = 1e-3
HIDDEN_DIM = 256
VAL_RATIO = 0.2
# SEED = 91643

# === Reproducibility ===
SEED = random.randint(1, 99999)
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# === Reproducibility ===
# torch.manual_seed(SEED)
# random.seed(SEED)
# np.random.seed(SEED)

def build_label_map(Y):
    labels = sorted(set(Y))
    return {label: i for i, label in enumerate(labels)}, {i: label for i, label in enumerate(labels)}

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
    return train_set, val_set, label2id, id2label, META

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

def train():
    train_set, val_set, label2id, id2label, meta_all = load_data(PT_FILE)
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

    torch.save(model.state_dict(), MODEL_FILE)

    # Plot
    # plt.figure()
    # plt.plot([log["epoch"] for log in logs], [log["loss"] for log in logs], label="Loss")
    # plt.plot([log["epoch"] for log in logs], [log["accuracy"] for log in logs], label="Accuracy")
    # plt.plot([log["epoch"] for log in logs], [log["f1"] for log in logs], label="Macro F1")
    # plt.xlabel("Epoch")
    # plt.ylabel("Metric")
    # plt.legend()
    # plt.grid(True)
    # plt.title("Training Curve")
    # plt.savefig(PLOT_FILE)
    # plt.close()
    # === Dual Y-Axis Plot (Left: Accuracy/F1, Right: Loss) — Matches "Narrative GCN" Style
    epochs = [log["epoch"] for log in logs]
    losses = [log["loss"] for log in logs]
    accs = [log["accuracy"] for log in logs]
    f1s = [log["f1"] for log in logs]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left Y-axis: Accuracy / F1
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy / F1", color="orangered")
    l1 = ax1.plot(epochs, accs, label="Accuracy", color="orange", linewidth=2)
    l2 = ax1.plot(epochs, f1s, label="Macro-F1", color="green", linewidth=2)
    ax1.tick_params(axis="y", labelcolor="orangered")
    ax1.set_ylim(0, 1)

    # Right Y-axis: Loss
    ax2 = ax1.twinx()
    ax2.set_ylabel("Loss", color="blue")
    l3 = ax2.plot(epochs, losses, label="Loss", color="blue", linewidth=2)
    ax2.tick_params(axis="y", labelcolor="blue")
    # Optional: customize y-limits for clearer contrast
    # ax2.set_ylim(min(losses), max(losses) + 10)

    # Combine all lines for legend
    lines = l1 + l2 + l3
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="lower right")

    plt.title("MSG-MLP Training Performance")
    fig.tight_layout()
    plt.grid(True)
    plt.savefig(PLOT_FILE)
    plt.close()




    # Save log JSON
    with open(TRAIN_LOG_FILE, "w") as f:
        json.dump({
            "log": logs,
            "seed": SEED,
            "input_file": os.path.basename(PT_FILE),
            "label_type": "subcode",
            "label2id": label2id
        }, f, indent=2)

    # Save errors
    val_X = torch.stack([val_set[i][0] for i in range(len(val_set))])
    val_Y = torch.tensor([val_set[i][1] for i in range(len(val_set))])
    val_indices = val_set.indices
    with torch.no_grad():
        logits = model(val_X)
        preds = logits.argmax(dim=1).tolist()

    errors = []
    for idx, pred, true in zip(val_indices, preds, val_Y.tolist()):
        if pred != true:
            errors.append({
                "graph_id": meta_all[idx]["graph_id"],
                "center_id": meta_all[idx]["center_id"],
                "original_text": meta_all[idx]["original_text"],
                "true_label": id2label[true],
                "pred_label": id2label[pred]
            })
    with open(TRAIN_ERR_FILE, "w") as f:
        json.dump(errors, f, indent=2)

    # === TESTING after training ===
    print("\n🔍 Evaluating MSG-MLP on Full Data (Post-Training)...")
    data = torch.load(PT_FILE)
    X_raw, Y_raw, META = data["X"], data["Y"], data["meta"]
    label2id, id2label = build_label_map(Y_raw)

    model.eval()
    X = torch.stack(X_raw)
    Y = torch.tensor([label2id[y] for y in Y_raw])
    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=1).tolist()

    acc = accuracy_score(Y, preds)
    f1 = f1_score(Y, preds, average="macro")

    print(f"✅ MSG-MLP Test Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")

    from sklearn.metrics import classification_report
    report = classification_report(Y, preds, target_names=[id2label[i] for i in sorted(id2label)], digits=4)
    print("\n=== Classification Report (Test Set) ===")
    print(report)

    # Save report summary
    with open(os.path.join(OUTPUT_DIR, "narrative_ego_mlp_test_log.json"), "w") as f:
        json.dump({
            "accuracy": round(acc, 6),
            "macro_f1": round(f1, 6),
            "support": len(Y),
            "label_type": "subcode"
        }, f, indent=2)

    # Save errors
    errors = []
    for i, (p, t) in enumerate(zip(preds, Y.tolist())):
        if p != t:
            errors.append({
                "graph_id": META[i]["graph_id"],
                "center_id": META[i]["center_id"],
                "original_text": META[i]["original_text"],
                "true_label": id2label[t],
                "pred_label": id2label[p]
            })
    with open(os.path.join(OUTPUT_DIR, "narrative_ego_mlp_test_errors.json"), "w") as f:
        json.dump(errors, f, indent=2)


# def test_only():
#     data = torch.load(PT_FILE)
#     X_raw, Y_raw, META = data["X"], data["Y"], data["meta"]
#     label2id, id2label = build_label_map(Y_raw)
#     model = MLPClassifier(768, HIDDEN_DIM, len(label2id))
#     model.load_state_dict(torch.load(MODEL_FILE))
#     model.eval()

#     X = torch.stack(X_raw)
#     Y = torch.tensor([label2id[y] for y in Y_raw])
#     preds = []
#     with torch.no_grad():
#         out = model(X)
#         preds = out.argmax(dim=1).tolist()

#     correct = [int(p == y) for p, y in zip(preds, Y.tolist())]
#     acc = sum(correct) / len(Y)
#     f1 = f1_score(Y, preds, average="macro")

#     with open(TEST_LOG_FILE, "w") as f:
#         json.dump({
#             "accuracy": round(acc, 6),
#             "macro_f1": round(f1, 6),
#             "support": len(Y),
#             "label_type": "subcode"
#         }, f, indent=2)

#     # Save errors
#     errors = []
#     for i, (p, t) in enumerate(zip(preds, Y.tolist())):
#         if p != t:
#             errors.append({
#                 "graph_id": META[i]["graph_id"],
#                 "center_id": META[i]["center_id"],
#                 "original_text": META[i]["original_text"],
#                 "true_label": id2label[t],
#                 "pred_label": id2label[p]
#             })
#     with open(TEST_ERR_FILE, "w") as f:
#         json.dump(errors, f, indent=2)

#     print(f"✅ Test Accuracy: {acc:.4f} | Macro F1: {f1:.4f} | Errors saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-only", action="store_true", help="Run evaluation only")
    args = parser.parse_args()

    if args.test_only:
        test_only()
    else:
        train()
