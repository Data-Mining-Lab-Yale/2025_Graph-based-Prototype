import os
import json

# Paths
# GRAPH_FOLDER = "path_to_your_graph_folder"
# GRAPH_FOLDER = "filtered_amr_graphs"
# GRAPH_FOLDER = "filtered_dependency_graphs"
# GRAPH_FOLDER = "filtered_srl_graphs_anchored"
# GRAPH_FOLDER = "filtered_srl_graphs_predicate"
GRAPH_FOLDER = "filtered_srl_graphs_weighted"



MAPPING_FILE = "EPPC_output_json/annotation_code_mapping_detailed_corrected.json"
# OUTPUT_FOLDER = "updated_amr_graphs"  # Will be created if not exist
# OUTPUT_FOLDER = "updated_dependency_graphs"  # Will be created if not exist
# OUTPUT_FOLDER = "updated_srl_graphs_anchored"  # Will be created if not exist
# OUTPUT_FOLDER = "updated_srl_graphs_predicate"  # Will be created if not exist
OUTPUT_FOLDER = "updated_srl_graphs_weighted"  # Will be created if not exist

# Load annotation-to-codebook mapping
with open(MAPPING_FILE, "r") as f:
    mapping = json.load(f)

# Create output directory
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Process all .json graph files
for filename in os.listdir(GRAPH_FOLDER):
    if not filename.endswith(".json"):
        continue

    file_path = os.path.join(GRAPH_FOLDER, filename)

    with open(file_path, "r", encoding="utf-8") as f:
        graph = json.load(f)

    annotation_label = graph.get("label")
    if annotation_label in mapping:
        graph["label"] = mapping[annotation_label]["matched_codebook_label"]
    else:
        print(f"[Warning] Label '{annotation_label}' in {filename} not found in mapping.")

    # Save updated graph
    output_path = os.path.join(OUTPUT_FOLDER, filename)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)

print("✅ Label update complete. Files saved to:", OUTPUT_FOLDER)
