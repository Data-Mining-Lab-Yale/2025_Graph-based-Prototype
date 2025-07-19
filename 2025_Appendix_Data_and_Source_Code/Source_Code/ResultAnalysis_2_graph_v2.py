import json
import pandas as pd
from pathlib import Path

# === CONFIG: Your confirmed result folders ===
RESULT_FOLDERS = [
    "results/results_sentence",
    "results/results_subsentence",
    "results/results_dep_gcn",
    "results/results_srl_anchored",
    "results/results_srl_predicate",
    "results/results_srl_weighted",
    "results/results_srl_gcn_weighted",
    "results/results_amr_gcn"
]

OUTPUT_CSV = "comparison_summary_best_f1.csv"

# === Load the best epoch from any *_train_log.json in the folder ===
def load_best_epoch_metrics_from_folder(folder: Path):
    try:
        log_files = list(folder.glob("*_train_log.json"))
        if not log_files:
            print(f"⚠️ No *_train_log.json file found in: {folder}")
            return None

        log_path = log_files[0]  # Load the first match
        with open(log_path, "r", encoding="utf-8") as f:
            data = json.load(f)

            # Accept either 'log' or 'metrics' as the log field
            log = data.get("log") or data.get("metrics") or []
            if not isinstance(log, list) or not log:
                print(f"⚠️ Invalid or missing log data in: {log_path}")
                return None

            best = max(log, key=lambda x: x.get("f1", -1))

            return {
                "Model": log_path.stem.replace("_train_log", ""),
                "Epoch": best["epoch"],
                "Accuracy (%)": round(best["accuracy"] * 100, 1),
                "F1 Score": round(best["f1"], 3),
                "Loss": round(best["loss"], 4)
            }

    except Exception as e:
        print(f"❌ Error reading from folder {folder}: {e}")
        return None

# === Collect all best results ===
results = []
for folder_name in RESULT_FOLDERS:
    folder = Path(folder_name)
    if not folder.exists():
        print(f"❌ Folder not found: {folder_name}")
        continue

    metrics = load_best_epoch_metrics_from_folder(folder)
    if metrics:
        results.append(metrics)

# === Output table ===
df = pd.DataFrame(results)
df = df.sort_values(by="F1 Score", ascending=False)

print("\n📊 Best F1 Performance by Model:\n")
print(df.to_string(index=False))

df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Saved summary to: {OUTPUT_CSV}")
