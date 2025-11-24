import json

# === Load input JSON ===
input_file = "EPPC_output_json/Labels/input_2_classified_intents_human_corrected.json"
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# === Prepare output containers ===
interactional = []
goal_oriented = []

# === Helper function to detect the level based on the key ===
def detect_level(section_name):
    return section_name  # "code", "subcode", "subsubcode"

# === Process each level ===
for section_name, items in data.items():
    for item in items:
        record = {
            "index": item["index"],
            "label": item["label"],
            "level": detect_level(section_name)
        }
        if item["type"] == "Interactional":
            interactional.append(record)
        elif item["type"] == "Goal-Oriented":
            goal_oriented.append(record)

# === Save split results ===
output = {
    "Interactional": interactional,
    "Goal-Oriented": goal_oriented
}

with open("EPPC_output_json/Labels/split_intents_by_type.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("Done! Saved to split_intents_by_type.json")
