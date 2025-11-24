import json
import os

def match_and_split_annotations(sentences_file, interactional_file, goal_oriented_file):
    """
    Reads a file with sentences and subsentences, and matches their spans to
    annotation data from two other files. It then saves the results to four
    separate JSON files, one for each combination of language unit and label type.
    """
    # 1. Load the data from the three files
    try:
        with open(sentences_file, 'r') as f:
            sentences_data = json.load(f)
        
        with open(interactional_file, 'r') as f:
            interactional_annotations = json.load(f)

        with open(goal_oriented_file, 'r') as f:
            goal_oriented_annotations = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: A file was not found. Please ensure all input files are in the same directory.")
        return
    except json.JSONDecodeError as e:
        print(f"Error: There was an issue decoding a JSON file. Please check its format.")
        return

    # 2. Create lookup tables from the annotation data
    # The key for the lookup is a tuple of (message_id, annotation_span_text) to handle non-unique spans
    interactional_lookup = {}
    for record in interactional_annotations:
        message_id = record.get("message_id")
        for annotation in record.get("annotations", []):
            key = (message_id, annotation.get("text"))
            if key not in interactional_lookup:
                interactional_lookup[key] = []
            interactional_lookup[key].append({
                "label": annotation.get("code", []),
                "label_type": annotation.get("label_type", [])
            })

    goal_oriented_lookup = {}
    for record in goal_oriented_annotations:
        message_id = record.get("message_id")
        for annotation in record.get("annotations", []):
            key = (message_id, annotation.get("text"))
            if key not in goal_oriented_lookup:
                goal_oriented_lookup[key] = []
            goal_oriented_lookup[key].append({
                "label": annotation.get("code", []),
                "label_type": annotation.get("label_type", [])
            })

    # 3. Initialize the output dictionaries
    sentence_interactional_output = {}
    subsentence_interactional_output = {}
    sentence_goal_oriented_output = {}
    subsentence_goal_oriented_output = {}

    # 4. Iterate through messages, sentences, and subsentences to find matches
    for message_record in sentences_data:
        message_id = message_record.get("message_id")
        for sentence_record in message_record.get("sentences", []):
            sentence_id = sentence_record.get("sentence_id")
            span_text = sentence_record.get("most_close_annotation_span", "")

            if span_text:
                lookup_key = (message_id, span_text)
                
                # Check for Interactional match
                if lookup_key in interactional_lookup:
                    sentence_interactional_output[sentence_id] = {
                        "text": sentence_record.get("sentence"),
                        "span": span_text,
                        "labels": interactional_lookup[lookup_key]
                    }

                # Check for Goal-Oriented match
                if lookup_key in goal_oriented_lookup:
                    sentence_goal_oriented_output[sentence_id] = {
                        "text": sentence_record.get("sentence"),
                        "span": span_text,
                        "labels": goal_oriented_lookup[lookup_key]
                    }

            for subsentence_record in sentence_record.get("subsentences", []):
                subsentence_id = subsentence_record.get("subsentence_id")
                span_text = subsentence_record.get("most_close_annotation_span", "")

                if span_text:
                    lookup_key = (message_id, span_text)
                    
                    # Check for Interactional match
                    if lookup_key in interactional_lookup:
                        subsentence_interactional_output[subsentence_id] = {
                            "text": subsentence_record.get("subsentence"),
                            "span": span_text,
                            "labels": interactional_lookup[lookup_key]
                        }

                    # Check for Goal-Oriented match
                    if lookup_key in goal_oriented_lookup:
                        subsentence_goal_oriented_output[subsentence_id] = {
                            "text": subsentence_record.get("subsentence"),
                            "span": span_text,
                            "labels": goal_oriented_lookup[lookup_key]
                        }

    # 5. Save the results to four separate JSON files
    output_files = {
        "EPPC_output_json/sentence_interactional_label.json": sentence_interactional_output,
        "EPPC_output_json/subsentence_interactional_label.json": subsentence_interactional_output,
        "EPPC_output_json/sentence_goal_oriented_label.json": sentence_goal_oriented_output,
        "EPPC_output_json/subsentence_goal_oriented_label.json": subsentence_goal_oriented_output
    }

    for filename, data in output_files.items():
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Successfully saved matched data to '{filename}'")

if __name__ == "__main__":
    match_and_split_annotations(
        sentences_file="EPPC_output_json/CleanedData/messages_with_sentences_and_subsentences.json",
        interactional_file="EPPC_output_json/CleanedData/data_1_interactional_annotations.json",
        goal_oriented_file="EPPC_output_json/CleanedData/data_1_goal_oriented_annotations.json"
    )