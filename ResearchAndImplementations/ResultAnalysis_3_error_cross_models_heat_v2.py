import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

# === Standardized error files ===
input_files = {
    "sentence": "results/results_sentence/sentence_errors_standardized.json",
    "subsentence": "results/results_subsentence/subsentence_errors_standardized.json",
    "dep-GCN": "results/results_dep_gcn/dep_gcn_errors_standardized.json",
    "srl-anchored": "results/results_srl_anchored/srl_anchored_errors_standardized.json",
    "srl-predicate": "results/results_srl_predicate/srl_predicate_errors_standardized.json",
    "srl-weighted": "results/results_srl_gcn_weighted/srl_gcn_weighted_errors_standardized.json",
    "amr-GCN": "results/results_amr_gcn/amr_gcn_errors_standardized.json",
    "narrative_amr-GCN":"results/results_narrative_ego_amr/573_errors_standardized.json",
    "narrative_MLP":"results/results_narrative_ego_mlp_3/narrative_ego_mlp_errors_standardized.json"
}

# === Step 1: Load error counts per true_label per model ===
label_model_error_counts = defaultdict(lambda: defaultdict(int))
all_labels = set()

for model_name, file_path in input_files.items():
    if not os.path.exists(file_path):
        print(f"⚠️ Missing file: {file_path}")
        continue

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"❌ Failed to load {file_path}: {e}")
            continue

    for item in data:
        true_label = item.get("true_label", "[missing]")
        label_model_error_counts[true_label][model_name] += 1
        all_labels.add(true_label)

sorted_labels = sorted(all_labels)

# === Step 2: Build error count dataframe ===
df = pd.DataFrame(index=sorted_labels, columns=input_files.keys()).fillna(0)

for label in sorted_labels:
    for model in input_files:
        df.loc[label, model] = label_model_error_counts[label][model]

# Step 3: Load total counts from annotation file
# Load gold annotation file
with open("EPPC_output_json/subsentence_subcode_labels.json", "r", encoding="utf-8") as f:
    try:
        gold_data = json.load(f)
        print(f"✅ File loaded. Top-level type: {type(gold_data)}")
        print(f"🔢 Total top-level keys: {len(gold_data)}")

        # Print first 3 keys and values
        for i, (k, v) in enumerate(gold_data.items()):
            print(f"[{i}] key: {k}, value: {v}")
            if i >= 2:
                break

    except Exception as e:
        print(f"❌ Failed to load file: {e}")
        gold_data = {}

# Correct total count extraction
# total_counts = defaultdict(int)
# for item in gold_data.values():
#     if isinstance(item, dict) and "subcode" in item:
#         label = item["subcode"]
#         norm_label = label.strip().replace(" ", "").lower()
#         total_counts[norm_label] += 1
total_counts = defaultdict(int)

for item in gold_data.values():
    if "labels" in item and isinstance(item["labels"], list):
        for label_entry in item["labels"]:
            if label_entry.get("level") == "subcode" and "label" in label_entry:
                raw_label = label_entry["label"]
                norm_label = raw_label.strip().replace(" ", "").lower()
                total_counts[norm_label] += 1


# DEBUG: Print top counts
print("\n🔍 Top label counts from gold annotation:")
sorted_total_counts = sorted(total_counts.items(), key=lambda x: x[1], reverse=True)
for label, count in sorted_total_counts[:10]:
    print(f"{label}: {count}")
print(f"... Total unique labels in gold annotation: {len(total_counts)}")


# Format index labels by matching normalized keys
def format_label(label):
    norm_label = label.strip().replace(" ", "").lower()
    return f"{label} ({total_counts.get(norm_label, 0)})"

# Apply label formatting to rows
df.index = [format_label(label) for label in df.index]

# === Step 5: Save CSVs ===
df.to_csv("error_counts_with_totals.csv")
# df_error_rate.to_csv("error_rates.csv")
print("📄 Saved: error_counts_with_totals.csv and error_rates.csv")

# === Step 6: Plot error count heatmap ===
plt.figure(figsize=(10, 12))
sns.heatmap(
    df[input_files.keys()].astype(int),
    annot=True, fmt='d',
    cmap="YlOrBr", linewidths=0.5, linecolor="gray",
    cbar_kws={"label": "Error Count"}
)
plt.title("True Label Error Counts Across Graph Models")
plt.xlabel("Model")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("error_analysis_heatmap_with_totals.png", dpi=300)
plt.show()

# # === Optional: Plot error rate heatmap ===
# plt.figure(figsize=(10, 12))
# sns.heatmap(
#     df_error_rate,
#     annot=True, fmt=".2f",
#     cmap="coolwarm", linewidths=0.5, linecolor="gray",
#     cbar_kws={"label": "Error Rate"}
# )
# plt.title("Error Rate Across Graph Models (Error / Total)")
# plt.xlabel("Model")
# plt.ylabel("True Label")
# plt.tight_layout()
# plt.savefig("error_rate_heatmap.png", dpi=300)
# plt.show()
