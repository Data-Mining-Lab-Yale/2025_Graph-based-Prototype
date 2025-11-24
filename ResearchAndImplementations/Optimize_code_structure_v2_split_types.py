import json
import os

def split_annotations_by_label_type(annotations_file, type_lookup_file):
    """
    Reads processed annotations and a label-to-type lookup file.
    Splits the annotations into two files, one for 'Interactional' and one for
    'Goal-Oriented' labels.
    """
    # 1. Load the data from the two files
    try:
        with open(annotations_file, 'r') as f:
            annotations_data = json.load(f)
        
        with open(type_lookup_file, 'r') as f:
            type_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: A file was not found. Please ensure both input files are in the same directory.")
        return
    except json.JSONDecodeError as e:
        print(f"Error: There was an issue decoding one of the JSON files. Please check the file's format.")
        return

    # 2. Create a helper dictionary to map each label to its type
    label_to_type_mapping = {}
    for label_type, labels in type_data.items():
        for item in labels:
            label_to_type_mapping[item["label"]] = label_type

    # 3. Process the annotations and split into two lists
    interactional_records = []
    goal_oriented_records = []

    for message_record in annotations_data:
        message_id = message_record.get("message_id")
        message_text = message_record.get("message")
        annotations = message_record.get("annotations", [])

        interactional_annotations = []
        goal_oriented_annotations = []
        
        for annotation in annotations:
            original_codes = annotation.get("code", [])
            
            # Filter labels based on their type using the lookup map
            interactional_codes = [code for code in original_codes if label_to_type_mapping.get(code) == "Interactional"]
            goal_oriented_codes = [code for code in original_codes if label_to_type_mapping.get(code) == "Goal-Oriented"]

            # If there are any interactional codes, create an annotation record for it
            if interactional_codes:
                new_annotation = annotation.copy()
                new_annotation["code"] = interactional_codes
                # The label_type will be ["Interactional"] since we've filtered the codes
                new_annotation["label_type"] = ["Interactional"]
                interactional_annotations.append(new_annotation)
            
            # If there are any goal-oriented codes, create an annotation record for it
            if goal_oriented_codes:
                new_annotation = annotation.copy()
                new_annotation["code"] = goal_oriented_codes
                # The label_type will be ["Goal-Oriented"]
                new_annotation["label_type"] = ["Goal-Oriented"]
                goal_oriented_annotations.append(new_annotation)

        # Create new records for each type if they have annotations
        if interactional_annotations:
            interactional_records.append({
                "message_id": message_id,
                "message": message_text,
                "annotations": interactional_annotations
            })

        if goal_oriented_annotations:
            goal_oriented_records.append({
                "message_id": message_id,
                "message": message_text,
                "annotations": goal_oriented_annotations
            })

    # 4. Save the results to two separate JSON files
    interactional_output_file = "EPPC_output_json/CleanedData/interactional_annotations.json"
    with open(interactional_output_file, 'w') as f:
        json.dump(interactional_records, f, indent=2)
    print(f"Successfully saved Interactional annotations to '{interactional_output_file}'")

    goal_oriented_output_file = "EPPC_output_json/CleanedData/goal_oriented_annotations.json"
    with open(goal_oriented_output_file, 'w') as f:
        json.dump(goal_oriented_records, f, indent=2)
    print(f"Successfully saved Goal-Oriented annotations to '{goal_oriented_output_file}'")

if __name__ == "__main__":
    split_annotations_by_label_type(
        annotations_file="EPPC_output_json/CleanedData/processed_annotations_with_types.json",
        type_lookup_file="EPPC_output_json/Labels/split_intents_by_type.json"
    )