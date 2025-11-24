import json
from collections import defaultdict
from pathlib import Path

# ===== CONFIG =====
INPUT_FILE = "EPPC_output_json/node_names_by_type_with_index.json"
OUTPUT_FILE = "EPPC_output_json/cleaned_node_names.json"
MAPPING_FILE = "EPPC_output_json/index_mapping.json"

# ===== LOAD =====
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Expected structure: { "code": [...], "subcode": [...], "subsubcode": [...] }
# Each entry like "1_1_2: Employment_Status"
mapping_changes = {}
cleaned_data = {"code": [], "subcode": [], "subsubcode": []}

# --- Copy codes and subcodes directly ---
cleaned_data["code"] = list(data.get("code", []))
cleaned_data["subcode"] = list(data.get("subcode", []))

# --- Group subsubcodes by parent <code>_<subcode> ---
groups = defaultdict(list)
for entry in data.get("subsubcode", []):
    if ":" not in entry:
        continue
    idx, label = entry.split(":", 1)
    idx = idx.strip()
    label = label.strip()

    # parent prefix like 1_1
    parts = idx.split("_")
    if len(parts) != 3:
        continue
    parent = f"{parts[0]}_{parts[1]}"
    groups[parent].append((idx, label))

# --- Process each group ---
for parent, entries in groups.items():
    seen_labels = {}
    new_index = 1
    for idx, label in entries:
        if label in seen_labels:
            # Duplicate — drop and record mapping
            mapping_changes[idx] = seen_labels[label]
        else:
            new_idx = f"{parent}_{new_index}"
            if idx != new_idx:
                mapping_changes[idx] = new_idx
            cleaned_data["subsubcode"].append(f"{new_idx}: {label}")
            seen_labels[label] = new_idx
            new_index += 1

# ===== SAVE =====
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

with open(MAPPING_FILE, "w", encoding="utf-8") as f:
    json.dump(mapping_changes, f, indent=2, ensure_ascii=False)

print(f"✅ Cleaned data saved to {OUTPUT_FILE}")
print(f"✅ Mapping of old → new indices saved to {MAPPING_FILE}")
