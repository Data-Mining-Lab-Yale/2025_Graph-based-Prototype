# INPUT_JSON = "EPPC_output_json/sentence_subcode_labels.json"  # or use subsentence_subcode_labels.json
# INPUT_JSON = "EPPC_output_json/subsentence_subcode_labels.json"
# Top of file
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import json, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
from datetime import datetime

# ========== CONFIG ==========
# INPUT_JSON = "EPPC_output_json/sentence_subcode_labels.json"  # ← change for clause
INPUT_JSON = "EPPC_output_json/subsentence_subcode_labels.json"
LABEL_MAPPING_JSON = "EPPC_output_json/annotation_code_mapping_detailed_corrected.json"
OUTPUT_DIR = "results_clause"  # ← change to "results_clause" for clause model
NUM_EPOCHS = 12
BATCH_SIZE = 16
MAX_LEN = 64
LEARNING_RATE = 2e-5
EARLY_STOPPING_PATIENCE = 3
# ============================

ERROR_LOG_PATH = os.path.join(OUTPUT_DIR, "errors.json")
COMPARISON_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "comparison_summary.json")

class TextDataset(Dataset):
    def __init__(self, samples, labels, tokenizer, max_len=64):
        self.samples = samples
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.samples[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx])
        }

class MLPClassifier(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        print("🔧 Initializing BERT model...")
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.fc(cls)

def load_label_mapping(path):
    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    return {
        k: v["matched_codebook_label"]
        for k, v in mapping.items()
        if v.get("level") == "subcode"
    }

def load_json_text_and_labels(path, label_map):
    print(f"📂 Loading dataset from {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts, labels, skipped = [], [], 0
    for entry in data.values():
        label = next((x["label"] for x in entry.get("labels", []) if x.get("level") == "subcode"), None)
        if label:
            texts.append(entry["text"])
            labels.append(label_map.get(label, label))
        else:
            skipped += 1
    print(f"✅ Loaded {len(texts)} samples with subcode labels. Skipped: {skipped}")
    return texts, labels

def evaluate(model, dataloader, device, texts=None, label_encoder=None):
    model.eval()
    all_preds, all_labels, logs = [], [], []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if texts and label_encoder:
                for j in range(len(labels)):
                    logs.append({
                        "text": texts[i * BATCH_SIZE + j],
                        "true_label": label_encoder.inverse_transform([labels[j].cpu().item()])[0],
                        "pred_label": label_encoder.inverse_transform([preds[j].cpu().item()])[0],
                        "correct": labels[j].cpu().item() == preds[j].cpu().item()
                    })

    if logs:
        with open(ERROR_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2)
        print(f"📝 Saved prediction errors to: {ERROR_LOG_PATH}")

    return accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average="macro"), all_labels, all_preds

def train():
    print("🚀 Starting training process...")

    label_map = load_label_mapping(LABEL_MAPPING_JSON)
    texts, raw_labels = load_json_text_and_labels(INPUT_JSON, label_map)

    print("🔤 Initializing BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(raw_labels)
    class_names = list(label_encoder.classes_)
    print(f"🔢 Classes: {len(class_names)} subcodes")

    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    train_data = TextDataset(X_train, y_train, tokenizer, max_len=MAX_LEN)
    test_data = TextDataset(X_test, y_test, tokenizer, max_len=MAX_LEN)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    model = MLPClassifier(hidden_dim=128, num_classes=len(class_names))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log = {
        "input_file": INPUT_JSON,
        "class_names": class_names,
        "metrics": []
    }

    best_f1 = 0
    for epoch in range(NUM_EPOCHS):
        print(f"\n🧠 Training model... Epoch {epoch+1}/{NUM_EPOCHS}")
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        acc, f1, y_true, y_pred = evaluate(model, test_loader, device, X_test, label_encoder)
        print(f"📉 Epoch {epoch+1} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

        log["metrics"].append({"epoch": epoch + 1, "loss": avg_loss, "accuracy": acc, "f1": f1})
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))
            print(f"💾 New best model saved to: {os.path.join(OUTPUT_DIR, 'best_model.pt')}")

    # Save log and plot
    with open(os.path.join(OUTPUT_DIR, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"📊 Metrics saved to: {os.path.join(OUTPUT_DIR, 'train_log.json')}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot([m["epoch"] for m in log["metrics"]], [m["loss"] for m in log["metrics"]], label="Loss")
    plt.plot([m["epoch"] for m in log["metrics"]], [m["accuracy"] for m in log["metrics"]], label="Accuracy")
    plt.plot([m["epoch"] for m in log["metrics"]], [m["f1"] for m in log["metrics"]], label="Macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.legend()
    plt.title("Training Progress")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_plot.png"))
    print(f"📈 Saved training plot to: {os.path.join(OUTPUT_DIR, 'training_plot.png')}")

    # Compare if both runs exist
    other_dir = "results_clause" if OUTPUT_DIR.endswith("sentence") else "results_sentence"
    other_error_file = os.path.join(other_dir, "errors.json")
    if os.path.exists(other_error_file):
        print("🔍 Comparing predictions between models...")
        with open(other_error_file, "r", encoding="utf-8") as f1, open(ERROR_LOG_PATH, "r", encoding="utf-8") as f2:
            base = {e["text"]: e for e in json.load(f1)}
            current = {e["text"]: e for e in json.load(f2)}

        comparison = []
        for text in set(base) & set(current):
            s, c = base[text], current[text]
            outcome = (
                "Both Correct" if s["correct"] and c["correct"] else
                "Only Sentence Correct" if s["correct"] and not c["correct"] else
                "Only Clause Correct" if not s["correct"] and c["correct"] else
                "Both Wrong"
            )
            comparison.append({
                "text": text,
                "true_label": s["true_label"],
                "sentence_pred": s["pred_label"],
                "clause_pred": c["pred_label"],
                "outcome": outcome
            })

        summary = {}
        for row in comparison:
            summary[row["outcome"]] = summary.get(row["outcome"], 0) + 1

        with open(COMPARISON_OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "details": comparison}, f, indent=2)
        print(f"📊 Comparison saved to: {COMPARISON_OUTPUT_PATH}")

if __name__ == "__main__":
    train()
