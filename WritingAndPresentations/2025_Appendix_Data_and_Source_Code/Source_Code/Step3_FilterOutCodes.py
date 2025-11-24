import json
from collections import defaultdict

FOLDER = "EPPC_output_json/"

# Load structured label files
with open(FOLDER+"sentence_label_structured.json", "r", encoding="utf-8") as f:
    sentence_data = json.load(f)

with open(FOLDER+"subsentence_label_structured.json", "r", encoding="utf-8") as f:
    subsentence_data = json.load(f)

# Helper: split labels by level
def split_by_level(data, unit_type):
    split = {
        "code": {},
        "subcode": {},
        "subsubcode": {}
    }

    for uid, entry in data.items():
        text = entry["text"]
        span = entry["span"]
        for label in entry["labels"]:
            level = label.get("level", "unknown")
            if level in split:
                if uid not in split[level]:
                    split[level][uid] = {
                        "text": text,
                        "span": span,
                        "labels": []
                    }
                split[level][uid]["labels"].append(label)
    
    # Save files
    for level, examples in split.items():
        fname = f"{FOLDER}{unit_type}_{level}_labels.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)

    return split

# Process both sentence and subsentence
sentence_split = split_by_level(sentence_data, "sentence")
subsentence_split = split_by_level(subsentence_data, "subsentence")

# Check if any subsubcode exists
subsub_in_sentence = any(sentence_split["subsubcode"])
subsub_in_subsentence = any(subsentence_split["subsubcode"])

print("Subsubcode exists in sentence data:", subsub_in_sentence)
print("Subsubcode exists in subsentence data:", subsub_in_subsentence)
print("Files saved for code/subcode/subsubcode splits.")
