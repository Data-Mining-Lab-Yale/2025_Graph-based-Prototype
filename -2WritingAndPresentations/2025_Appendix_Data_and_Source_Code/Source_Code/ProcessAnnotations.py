import json

# === Input and Output File Paths ===
input_path = "Data/EPPC_sentence_dataset_0505_merge.json"
output_path = "EPPC_output_json/processed_messages_with_annotations.json"

# === Load the data ===
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# === Helper to find context chains ===
visited = set()
message_id = 0
messages = []

for idx, item in enumerate(data):
    if idx in visited or item["prev"] is not None:
        continue  # Only start from the beginning of a message

    context_text = item["context"]
    annotations = item.get("annotations", [])
    visited.add(idx)

    # Forward traversal to find full message
    current_item = item
    while current_item.get("next"):
        next_found = False
        for j, next_item in enumerate(data):
            if next_item["context"] == current_item["next"] and j not in visited:
                context_text += " " + next_item["context"]
                annotations += next_item.get("annotations", [])
                visited.add(j)
                current_item = next_item
                next_found = True
                break
        if not next_found:
            break

    # Build annotation list with text_id
    annots = []
    for i, ann in enumerate(annotations):
        annots.append({
            "text_id": f"{message_id}_{i}",
            "text": ann["text"],
            "code": ann["codes"]
        })

    # Save message entry
    messages.append({
        "message": context_text.strip(),
        "message_id": message_id,
        "annotations": annots
    })

    message_id += 1

# === Write output ===
with open(output_path, "w", encoding="utf-8") as f_out:
    json.dump(messages, f_out, indent=2, ensure_ascii=False)

print(f"✅ Output saved to {output_path}")
