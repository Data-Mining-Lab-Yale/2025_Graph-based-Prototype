# patience = 10
# best_f1 = 0
# epochs_without_improvement = 0

# INPUT_JSON = "EPPC_output_json/sentence_subcode_labels.json"  # or use subsentence_subcode_labels.json
# INPUT_JSON = "EPPC_output_json/subsentence_subcode_labels.json"

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
from collections import Counter

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
# INPUT_JSON = "EPPC_output_json/sentence_subcode_labels.json"
INPUT_JSON = "EPPC_output_json/subsentence_subcode_labels.json"
LABEL_TYPE = "subcode"
# SOURCE_TAG = "sentence"
SOURCE_TAG = "subsentence"
MODEL_NAME = "MLP"
# OUTPUT_DIR = "results_sentence"
OUTPUT_DIR = "results_subsentence"

os.makedirs(OUTPUT_DIR, exist_ok=True)

ERROR_LOG_PATH = os.path.join(OUTPUT_DIR, "errors.json")
TRAIN_LOG_PATH = os.path.join(OUTPUT_DIR, "train_log.json")
COMPARISON_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "comparison_summary.json")

NUM_EPOCHS = 120
BATCH_SIZE = 16
MAX_LEN = 64
LEARNING_RATE = 2e-5
PATIENCE = 10

# ==== DATA LOADING ====
def load_json_text_and_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts, labels, skipped = [], [], 0
    for uid, entry in data.items():
        label = None
        for ann in entry.get("labels", []):
            if ann.get("level") == LABEL_TYPE:
                label = ann["label"]
                break
        if label:
            texts.append(entry["text"])
            labels.append(label)
        else:
            skipped += 1
    print(f"✅ Loaded {len(texts)} samples with {LABEL_TYPE} labels. Skipped: {skipped}")
    return texts, labels

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.classifier(x)

# ==== EVALUATE ====
def evaluate(model, dataloader, tokenizer, label_encoder, text_list):
    model.eval()
    all_preds, all_labels, logs = [], [], []
    with torch.no_grad():
        for texts, labels in dataloader:
            enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
            input_ids = enc["input_ids"].cuda()
            attention_mask = enc["attention_mask"].cuda()
            labels = labels.cuda()
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state.mean(dim=1)
            logits = model(pooled)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            for j in range(len(texts)):
                logs.append({
                    "text": texts[j],
                    "true_label": label_encoder.inverse_transform([labels[j].cpu().item()])[0],
                    "pred_label": label_encoder.inverse_transform([preds[j].cpu().item()])[0],
                    "correct": labels[j].cpu().item() == preds[j].cpu().item()
                })

    with open(ERROR_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "model": MODEL_NAME,
            "source": SOURCE_TAG,
            "label_type": LABEL_TYPE,
            "seed": SEED,
            "errors": logs
        }, f, indent=2)
    print(f"📝 Saved prediction errors to: {ERROR_LOG_PATH}")
    return accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average="macro")

# ==== TRAIN ====
def train():
    print("🚀 Starting training process...")
    texts, labels = load_json_text_and_labels(INPUT_JSON)

    # Filter rare labels
    label_counts = Counter(labels)
    filtered = [(t, l) for t, l in zip(texts, labels) if label_counts[l] > 1]
    texts, labels = zip(*filtered)
    print(f"✅ Filtered rare labels: now using {len(texts)} samples across {len(set(labels))} classes")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    counts = Counter(y)
    if min(counts.values()) < 2:
        print("⚠️ Some classes have <2 samples. Disabling stratified split.")
        X_train, X_val, y_train, y_val = train_test_split(
            texts, y, test_size=0.2, random_state=SEED)
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            texts, y, test_size=0.2, stratify=y, random_state=SEED)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    global bert_model
    bert_model = BertModel.from_pretrained("bert-base-uncased").cuda()
    for param in bert_model.parameters():
        param.requires_grad = True

    train_loader = DataLoader(TextDataset(X_train, torch.tensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TextDataset(X_val, torch.tensor(y_val)), batch_size=BATCH_SIZE)

    model = MLPClassifier(768, len(label_encoder.classes_)).cuda()
    optimizer = torch.optim.Adam(
        list(bert_model.parameters()) + list(model.parameters()), lr=LEARNING_RATE
    )
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0
    epochs_without_improvement = 0
    metrics = []
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0
        for texts, labels in train_loader:
            enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
            input_ids = enc["input_ids"].cuda()
            attention_mask = enc["attention_mask"].cuda()
            labels = labels.cuda()
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state.mean(dim=1)
            logits = model(pooled)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        acc, f1 = evaluate(model, val_loader, tokenizer, label_encoder, X_val)
        avg_loss = total_loss / len(train_loader)
        metrics.append({"epoch": epoch, "loss": avg_loss, "accuracy": acc, "f1": f1})
        print(f"📉 Epoch {epoch:02d} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))
            print(f"💾 New best model saved to: {OUTPUT_DIR}/best_model.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print("⏹️ Early stopping triggered.")
                break

    # Save log
    train_log = {
        "model": MODEL_NAME,
        "source": SOURCE_TAG,
        "label_type": LABEL_TYPE,
        "seed": SEED,
        "input_file": INPUT_JSON,
        "class_names": list(label_encoder.classes_),
        "metrics": metrics
    }
    with open(TRAIN_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(train_log, f, indent=2)
    print(f"📊 Metrics saved to: {TRAIN_LOG_PATH}")

    # Plot
    # plt.figure()
    # plt.plot([m["epoch"] for m in metrics], [m["loss"] for m in metrics], label="Loss")
    # plt.plot([m["epoch"] for m in metrics], [m["f1"] for m in metrics], label="F1")
    # plt.xlabel("Epoch")
    # plt.ylabel("Metric")
    # plt.title("Training Performance")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(OUTPUT_DIR, "training_plot.png"))
    # plt.close()
    # print(f"📈 Saved training plot to: {OUTPUT_DIR}/training_plot.png")

    # Plot
    plt.figure()
    epochs = [m["epoch"] for m in metrics]
    losses = [m["loss"] for m in metrics]
    f1s = [m["f1"] for m in metrics]
    accs = [m["accuracy"] for m in metrics]

    plt.plot(epochs, losses, label="Loss")
    plt.plot(epochs, accs, label="Accuracy")
    plt.plot(epochs, f1s, label="F1")

    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Training Performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_plot.png"))
    plt.close()
    print(f"📈 Saved training plot to: {OUTPUT_DIR}/training_plot.png")

if __name__ == "__main__":
    train()
