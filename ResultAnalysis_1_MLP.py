import json
import pandas as pd

# === CONFIG: Paths to training log files ===
sentence_log_path = "results_sentence/sentence_train_log.json"
clause_log_path = "results_clause/subsentence_train_log.json"

# === Load training logs ===
with open(sentence_log_path, "r", encoding="utf-8") as f:
    sentence_log = json.load(f)

with open(clause_log_path, "r", encoding="utf-8") as f:
    clause_log = json.load(f)

# === Extract best (last epoch) metrics ===
best_sentence = sentence_log["metrics"][-1]
best_clause = clause_log["metrics"][-1]

# === Build results table ===
df = pd.DataFrame([
    {
        "Model": "MLP",
        "Graph Type": "None",
        "Source": "--",
        "Node Level": "Sentence",
        "Accuracy (%)": round(best_sentence["accuracy"] * 100, 1),
        "F1 Score": round(best_sentence["f1"], 3)
    },
    {
        "Model": "MLP",
        "Graph Type": "None",
        "Source": "--",
        "Node Level": "Sub-sentence",
        "Accuracy (%)": round(best_clause["accuracy"] * 100, 1),
        "F1 Score": round(best_clause["f1"], 3)
    }
])

# === Display and Save ===
print("📊 MLP Classification Performance:\n")
print(df.to_string(index=False))

# Optionally save to CSV
df.to_csv("mlp_comparison_summary.csv", index=False)
print("\n✅ Saved summary to: mlp_comparison_summary.csv")
