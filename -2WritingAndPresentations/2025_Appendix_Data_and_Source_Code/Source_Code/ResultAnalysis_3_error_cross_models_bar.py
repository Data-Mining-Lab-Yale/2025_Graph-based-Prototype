import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

# 📂 Folders containing *_errors.json
folders = [
    "results/results_sentence",
    "results/results_subsentence",
    "results/results_dep_gcn",
    "results/results_srl_anchored",
    "results/results_srl_predicate",
    "results/results_srl_weighted",
    "results/results_srl_gcn_weighted",
    "results/results_amr_gcn"
]

model_errors = {}  # model_name -> Counter of true_label

# 🔍 Collect errors
for folder in folders:
    error_file = None
    for file in os.listdir(folder):
        if file.endswith("_errors.json"):
            error_file = os.path.join(folder, file)
            break
    if error_file is None:
        continue

    model_name = os.path.basename(error_file).replace("_errors.json", "")
    with open(error_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    counts = Counter()
    for entry in data:
        if isinstance(entry, dict):
            true_label = entry.get("true_label")
            if true_label:
                counts[true_label] += 1
    model_errors[model_name] = counts

# 🧱 Build label space
all_labels = set()
for c in model_errors.values():
    all_labels.update(c.keys())
all_labels = sorted(list(all_labels))

# 📊 Plot
fig, ax = plt.subplots(figsize=(10, 0.5 * len(all_labels)))
width = 0.8 / len(model_errors)
colors = plt.cm.tab10.colors
label_totals = defaultdict(int)

for i, (model, counter) in enumerate(model_errors.items()):
    values = [counter.get(label, 0) for label in all_labels]
    for label, v in zip(all_labels, values):
        label_totals[label] += v
    ax.barh(
        [y + i * width for y in range(len(all_labels))],
        values,
        height=width,
        label=model,
        color=colors[i % len(colors)]
    )

# 🏷️ Label formatting
ax.set_yticks([y + width * (len(model_errors) / 2 - 0.5) for y in range(len(all_labels))])
ax.set_yticklabels([f"{label} ({label_totals[label]})" for label in all_labels])
ax.set_xlabel("Misclassified Instances")
ax.set_title("🔎 Error Count per Class across Models")
ax.legend()
plt.tight_layout()
plt.savefig("error_analysis_across_models.png", dpi=300)
plt.show()

# 📦 Save counts to JSON
output_counts = {
    model: dict(counter) for model, counter in model_errors.items()
}
with open("error_analysis_counts.json", "w", encoding="utf-8") as f:
    json.dump(output_counts, f, indent=2)
