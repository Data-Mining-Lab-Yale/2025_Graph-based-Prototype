import json

def fix_correct_field(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    corrected = []
    for entry in data:
        entry["correct"] = entry["true_label"] == entry["pred_label"]
        corrected.append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(corrected, f, indent=2)

# Example usage
fix_correct_field(
    "results/results_amr_gcn/amr_gcn_errors_standardized.json",
    "results/results_amr_gcn/amr_gcn_errors_corrected.json"
)
