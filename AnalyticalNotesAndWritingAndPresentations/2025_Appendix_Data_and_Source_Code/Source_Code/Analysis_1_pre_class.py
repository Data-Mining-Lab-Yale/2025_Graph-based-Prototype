import json
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import matplotlib.pyplot as plt
import os

# ==== Settings ====
# results_dep_gcn/
# results_sentence/
# results_clause/
ERROR_FILE = "results_dep_gcn/dep_gcn_errors.json"  # change to subsentence_errors.json or sentence_errors.json as needed
TRAIN_LOG_FILE = "results_dep_gcn/dep_gcn_train_log.json"
OUTPUT_DIR = "results_dep_gcn/analysis_dep_gcn"  # will be created if not exist

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==== Load data ====
with open(ERROR_FILE, "r", encoding="utf-8") as f:
    errors = json.load(f)

# with open(TRAIN_LOG_FILE, "r", encoding="utf-8") as f:
#     train_log = json.load(f)

# class_names = train_log["class_names"] if isinstance(train_log, dict) else train_log[0]["class_names"]
# label2id = {name: idx for idx, name in enumerate(class_names)}

# === Derive class names from the error file instead of train log ===
label_set = set()
for e in errors:
    label_set.add(e["true_label"])
    label_set.add(e["pred_label"])

class_names = sorted(label_set)
label2id = {name: idx for idx, name in enumerate(class_names)}


# ==== Prepare ground truth and predictions ====
y_true = []
y_pred = []

for e in errors:
    if e["true_label"] not in label2id or e["pred_label"] not in label2id:
        continue
    y_true.append(label2id[e["true_label"]])
    y_pred.append(label2id[e["pred_label"]])

# ==== Compute per-class scores ====
precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, labels=range(len(class_names)), zero_division=0
)

df = pd.DataFrame({
    "Class": class_names,
    "Support": support,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
}).sort_values(by="F1 Score", ascending=False)

# ==== Save CSV and plot ====
df.to_csv(os.path.join(OUTPUT_DIR, "per_class_f1.csv"), index=False)

plt.figure(figsize=(10, 6))
plt.barh(df["Class"], df["F1 Score"], color="teal")
plt.xlabel("F1 Score")
plt.title("Per-Class F1 Score")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "per_class_f1_plot.png"), dpi=300)
plt.close()

print("✅ Per-class F1 analysis completed. Results saved to:", OUTPUT_DIR)
