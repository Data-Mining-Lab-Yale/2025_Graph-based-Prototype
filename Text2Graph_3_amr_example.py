import amrlib
import penman
import networkx as nx
import matplotlib.pyplot as plt

# Load the AMR parser (STOG) model from local path
print("Loading AMR parser from local model path...")
model_path = "Models/model_parse_xfm_bart_large"
stog = amrlib.load_stog_model(model_dir=model_path)

# Example sentence
sentence = "Dr. Smith sent a PET scan order."

# Run parsing
print(f"\nParsing: {sentence}")
graphs = stog.parse_sentences([sentence])

# Output PENMAN graph
penman_str = graphs[0]
print("\n[PENMAN Graph]\n", penman_str)

# Parse PENMAN string to extract graph
graph = penman.decode(penman_str)
triples = graph.triples
print("\n[Triples]\n", triples)

# --- Build and visualize with NetworkX ---
G = nx.DiGraph()

# Add nodes and edges from triples
for subj, role, obj in triples:
    G.add_node(subj)
    G.add_node(obj)
    G.add_edge(subj, obj, label=role)

# Draw graph
pos = nx.spring_layout(G)
plt.figure(figsize=(10, 6))
nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=500)
nx.draw_networkx_labels(G, pos, font_size=10)
nx.draw_networkx_edges(G, pos, arrows=True)
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d["label"] for u, v, d in G.edges(data=True)}, font_color="red")
plt.title("AMR Graph from Sentence")
plt.axis("off")
plt.tight_layout()
plt.show()
