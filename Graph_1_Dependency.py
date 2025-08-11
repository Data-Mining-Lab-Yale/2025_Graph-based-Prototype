import json
import os
import spacy
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

# INPUT = "EPPC_output_json/sentence_goal_oriented_label.json"
# INPUT = "EPPC_output_json/sentence_interactional_label.json"
# INPUT = "EPPC_output_json/subsentence_goal_oriented_label.json"
INPUT = "EPPC_output_json/subsentence_interactional_label.json"


# Load the spaCy English model
# Run `python -m spacy download en_core_web_sm` in your terminal if you haven't already
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def create_dependency_graph(text):
    """
    Analyzes text using spaCy to create a dependency graph in a dictionary format.
    The format is compatible with the provided 0_0_0.json example.
    """
    doc = nlp(text)
    nodes = []
    links = []
    
    # Create nodes from tokens
    for i, token in enumerate(doc):
        nodes.append({
            "label": token.text,
            "pos": token.pos_,
            "id": i
        })

    # Create links from dependencies
    for token in doc:
        # Link children to their head
        for child in token.children:
            links.append({
                "dep": child.dep_,
                "source": token.i,
                "target": child.i
            })
    
    return {
        "directed": True,
        "multigraph": False,
        "graph": {},
        "nodes": nodes,
        "links": links
    }


def draw_and_save_graph(graph_data, output_path, title):
    """
    Draws a dependency graph from a dictionary and saves it as a PNG image.
    """
    G = nx.DiGraph()
    
    # Add nodes
    for node in graph_data["nodes"]:
        G.add_node(node["id"], label=node["label"], pos=node["pos"])
    
    # Add edges
    for link in graph_data["links"]:
        G.add_edge(link["source"], link["target"], dep=link["dep"])

    # Define node positions for a cleaner layout
    pos = nx.planar_layout(G) 
    
    plt.figure(figsize=(10, 8))
    
    node_labels = {node: f"{data['label']}\n({data['pos']})" for node, data in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_color='black')
    
    edge_labels = nx.get_edge_attributes(G, 'dep')
    
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    plt.title(title)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Change this variable to process a different input file
    INPUT_DATA_FILE = INPUT
    
    # Automatically generate output folder names
    file_name_without_ext = os.path.splitext(os.path.basename(INPUT_DATA_FILE))[0]
    output_base_dir = Path(file_name_without_ext)
    output_json_dir = output_base_dir / "json"
    output_images_dir = output_base_dir / "images"

    # Create output directories if they don't exist
    output_json_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(INPUT_DATA_FILE, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{INPUT_DATA_FILE}' was not found.")
        exit()

    for item_id, item_data in data.items():
        text = item_data.get("text")
        if text:
            # Generate the graph
            graph_json = create_dependency_graph(text)
            
            # Define output paths
            json_file_path = output_json_dir / f"{item_id}.json"
            image_file_path = output_images_dir / f"{item_id}.png"
            
            # Save the JSON file
            with open(json_file_path, 'w') as f:
                json.dump(graph_json, f, indent=2)
            
            # Draw and save the graph image
            draw_and_save_graph(graph_json, image_file_path, f"Dependency Graph for {item_id}")
            
            print(f"✅ Processed {item_id}: saved to {json_file_path} and {image_file_path}")
        else:
            print(f"⚠️ Skipped {item_id} due to missing text.")