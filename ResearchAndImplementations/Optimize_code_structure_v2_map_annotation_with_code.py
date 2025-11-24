import json

def process_annotations(annotations_file, mapping_file, type_file):
    """
    Reads data from three JSON files, processes the annotations by updating labels
    and adding label types, and returns the processed data.
    """
    # 1. Load the data from the three files
    try:
        with open(annotations_file, 'r') as f:
            annotations_data = json.load(f)
        
        with open(mapping_file, 'r') as f:
            mapping_data = json.load(f)

        with open(type_file, 'r') as f:
            type_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: One of the files was not found. Please ensure the files are in the same directory as the script.")
        print(e)
        return None
    except json.JSONDecodeError as e:
        print(f"Error: There was an issue decoding one of the JSON files. Please check the file format.")
        print(e)
        return None

    # 2. Create helper dictionaries for quick lookups
    # Map old labels to new codebook labels
    label_mapping = {old_label: data["matched_codebook_label"] for old_label, data in mapping_data.items()}

    # Map codebook labels to their type (Interactional or Goal-Oriented)
    type_mapping = {}
    for intent_type, labels in type_data.items():
        for item in labels:
            type_mapping[item["label"]] = intent_type

    # 3. Process the annotations data
    for message_record in annotations_data:
        for annotation in message_record.get("annotations", []):
            new_codes = []
            label_types = []
            
            # Replace old codes with new ones from the mapping
            for code in annotation.get("code", []):
                new_code = label_mapping.get(code, code)
                new_codes.append(new_code)
                
                # Find the label type for the new code
                if new_code in type_mapping:
                    label_types.append(type_mapping[new_code])
            
            annotation["code"] = new_codes
            annotation["label_type"] = sorted(list(set(label_types)))

    return annotations_data

if __name__ == "__main__":
    processed_data = process_annotations(
        annotations_file="EPPC_output_json/CleanedData/processed_messages_with_annotations.json",
        mapping_file="EPPC_output_json/Labels/annotation_code_mapping_detailed_corrected.json",
        type_file="EPPC_output_json/Labels/split_intents_by_type.json"
    )

    if processed_data:
        # Save the processed data to a new JSON file
        output_file_name = "processed_annotations_with_types.json"
        with open(output_file_name, 'w') as outfile:
            json.dump(processed_data, outfile, indent=2)
        print(f"Processed data has been successfully saved to {output_file_name}")