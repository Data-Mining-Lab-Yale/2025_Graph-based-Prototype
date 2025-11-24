import json
from collections import defaultdict

FOLDER = "EPPC_output_json/"

# Load structured label files
with open(FOLDER+"sentence_label_structured.json", "r", encoding="utf-8") as f:
    sentence_data = json.load(f)

with open(FOLDER+"subsentence_label_structured.json", "r", encoding="utf-8") as f:
    subsentence_data = json.load(f)

# Helper function to count label levels
def count_levels(data):
    counts = defaultdict(int)
    for entry in data.values():
        for label in entry.get("labels", []):
            level = label.get("level", "unknown")
            counts[level] += 1
    return counts

# Count labels in both sets
sentence_counts = count_levels(sentence_data)
subsentence_counts = count_levels(subsentence_data)

# Display results
print("=== Sentence-Level Annotation Counts ===")
for level, count in sentence_counts.items():
    print(f"{level}: {count}")

print("\n=== Subsentence-Level Annotation Counts ===")
for level, count in subsentence_counts.items():
    print(f"{level}: {count}")
