import json
from pathlib import Path

# === File paths ===
remove_file = Path("EPPC_output_json/Labels/goal_remove_list.json")
input_files = [
    Path("EPPC_output_json/subsentence_goal_oriented_label.json"),
    Path("EPPC_output_json/sentence_goal_oriented_label.json")
]

# === Load remove list ===
with open(remove_file, "r", encoding="utf-8") as f:
    remove_data = json.load(f)

# Collect all labels to remove
remove_labels = set(entry["label"] for entry in remove_data.get("remove", []))
print("Labels to remove:", remove_labels)

# === Process each input file ===
for infile in input_files:
    with open(infile, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = {}
    removed_count = 0

    for key, entry in data.items():
        keep = True
        for label_entry in entry.get("labels", []):
            labels = label_entry.get("label", [])
            if any(lbl in remove_labels for lbl in labels):
                keep = False
                break
        if keep:
            new_data[key] = entry
        else:
            removed_count += 1

    # Save output
    outfile = infile.with_name(infile.stem + "_filtered.json")
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

    print(f"{infile.name}: removed {removed_count} entries, kept {len(new_data)}. Saved to {outfile.name}")
