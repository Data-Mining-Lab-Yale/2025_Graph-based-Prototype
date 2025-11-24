import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# -----------------------------
# CONFIGURATION
# -----------------------------
input_files = {
    "sentence": "results/results_sentence/sentence_errors.json",
    "subsentence": "results/results_subsentence/subsentence_errors.json",
    "dep-GCN": "results/results_dep_gcn/dep_gcn_errors.json",
    "srl-anchored": "results/results_srl_anchored/srl_anchored_errors.json",
    "srl-predicate": "results/results_srl_predicate/srl_predicate_errors.json",
    "srl-weighted": "results/results_srl_weighted/srl_weighted_errors.json",
    "amr-GCN": "results/results_amr_gcn/amr_gcn_errors.json",
}

mapping_path = "EPPC_output_json/annotation_code_mapping_detailed_corrected.json"
output_json = "true_label_error_count_matrix.json"
save_fig_path = "error_analysis_heatmap.png"

# -----------------------------
# LOAD LABEL MAPPING
# -----------------------------
with open(mapping_path, "r", encoding="utf-8") as f:
    mapping = json.load(f)
map_true_label = {k: v["matched_codebook_label"] for k, v in mapping.items()}

# -----------------------------
# LOAD AND COUNT ERRORS
# -----------------------------
error_counts = {}
label_totals = {}

for model_name, filepath in input_files.items():
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            errors = json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue

    for entry in errors:
        if not isinstance(entry, dict):
            continue
        true_label = entry.get("true_label")
        if true_label is None:
            continue

        mapped_label = map_true_label.get(true_label, true_label)

        # Count total for normalization
        label_totals[mapped_label] = label_totals.get(mapped_label, 0) + 1

        # Count errors
        if mapped_label not in error_counts:
            error_counts[mapped_label] = {}
        error_counts[mapped_label][model_name] = error_counts[mapped_label].get(model_name, 0) + 1

# -----------------------------
# PREPARE DATAFRAME
# -----------------------------
all_labels = sorted(error_counts.keys())
all_models = list(input_files.keys())

data = []
for label in all_labels:
    row = []
    for model in all_models:
        row.append(error_counts.get(label, {}).get(model, 0))
    data.append(row)

df = pd.DataFrame(data, index=[
    f"{label.replace('_', ' ')} ({label_totals[label]})" for label in all_labels
], columns=all_models)

# -----------------------------
# PLOT HEATMAP
# -----------------------------
plt.figure(figsize=(15, 11))
sns.set(font_scale=0.95)
ax = sns.heatmap(
    df, annot=False, cmap="YlOrBr", linewidths=0.5, linecolor="gray",
    cbar_kws={"label": "Error Count"}
)
plt.title("True Label Error Counts Across Graph Models (Incorrect Predictions Only)", fontsize=14)
plt.ylabel("True Label")
plt.xlabel("Model")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(save_fig_path, dpi=300)
plt.show()

# -----------------------------
# SAVE AS JSON
# -----------------------------
df.to_json(output_json, orient="index")
print(f"✅ Saved error matrix JSON: {output_json}")
print(f"✅ Saved plot: {save_fig_path}")
