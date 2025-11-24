import json
import os
from pathlib import Path
from tqdm import tqdm

# Define the models and their folders
error_paths = {
    "sentence": "results/results_sentence",
    "subsentence": "results/results_subsentence",
    "dep-GCN": "results/results_dep_gcn",
    "srl-anchored": "results/results_srl_anchored",
    "srl-predicate": "results/results_srl_predicate",
    "srl-weighted": "results/results_srl_weighted",
    "amr-GCN": "results/results_amr_gcn"
}

# Load the mapping file
with open("EPPC_output_json/annotation_code_mapping_detailed_corrected.json", "r") as f:
    mapping = json.load(f)

codebook_lookup = {
    key.lower(): value["matched_codebook_label"] for key, value in mapping.items()
}

# Output folder
output_dir = Path("formatted_errors")
output_dir.mkdir(exist_ok=True)

# Track skipped counts
skipped = {}

# Process each model
for model_name, folder in error_paths.items():
    model_id = model_name.replace("-", "_")
    error_file = Path(folder) / f"{model_id}_errors.json"
    
    if not error_file.exists():
        print(f"❌ File not found for model {model_name}: {error_file}")
        continue

    print(f"📄 Processing {model_name} from {error_file.name}")
    try:
        with open(error_file, "r", encoding="utf-8") as f:
            errors = json.load(f)
    except Exception as e:
        print(f"❌ Error loading {error_file}: {e}")
        continue

    aligned = []
    skipped[model_name] = 0

    for entry in tqdm(errors, desc=f"Aligning {model_name}"):
        # Some older files were saved as strings instead of dicts
        if not isinstance(entry, dict):
            skipped[model_name] += 1
            continue

        true_label = entry.get("true_label", "").strip().lower()
        pred_label = entry.get("predicted_label", "").strip().lower()

        # Apply mapping
        mapped_true = codebook_lookup.get(true_label)
        mapped_pred = codebook_lookup.get(pred_label)

        if not mapped_true or not mapped_pred:
            skipped[model_name] += 1
            continue

        aligned.append({
            "true_label": mapped_true,
            "predicted_label": mapped_pred,
            "text": entry.get("text", ""),
            "id": entry.get("id", None)
        })

    out_file = output_dir / f"{model_id}_errors.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(aligned, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(aligned)} aligned errors to {out_file} (Skipped: {skipped[model_name]})")

print("🎉 All done.")
