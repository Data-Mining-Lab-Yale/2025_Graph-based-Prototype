import json
import pandas as pd

# === CONFIG: Paths to training log files ===
sentence_log_path = "results_sentence/sentence_train_log.json"
clause_log_path = "results_subsentence/subsentence_train_log.json"
dep_gcn_log_path = "results_dep_gcn/dep_gcn_train_log.json"

# === Load training logs ===
with open(sentence_log_path, "r", encoding="utf-8") as f:
    sentence_log = json.load(f)

with open(clause_log_path, "r", encoding="utf-8") as f:
    clause_log = json.load(f)

with open(dep_gcn_log_path, "r", encoding="utf-8") as f:
    dep_gcn_log = json.load(f)    


# === Add DEP-GCN results ===
# with open("3000_dep_gcn_train_log.json", "r", encoding="utf-8") as f:
#     dep_gcn_log = json.load(f)
best_dep_gcn = max(dep_gcn_log["metrics"], key=lambda x: x["f1"])
# === Extract best (last epoch) metrics ===
best_sentence = sentence_log["metrics"][-1]
best_clause = clause_log["metrics"][-1]
best_dep_gcn = dep_gcn_log["metrics"][-1]

# === Build results table ===
df = pd.DataFrame([
    {
        "Model": "DEP-GCN",
        "Graph Type": "Dependency",
        "Source": "spaCy",
        "Node Level": "Token",
        "Accuracy (%)": round(best_dep_gcn["accuracy"] * 100, 1),
        "F1 Score": round(best_dep_gcn["f1"], 3)
    },
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
df.to_csv("comparison_summary.csv", index=False)
print("\n✅ Saved summary to: mlp_comparison_summary.csv")
