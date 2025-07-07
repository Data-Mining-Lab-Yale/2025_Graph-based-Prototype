import penman
from amrlib.models.model_factory import load_inference_model
import networkx as nx
import matplotlib.pyplot as plt

# === Config ===
custom_model_path = 'Models/model_parse_xfm_bart_large'
test_sentence = 'Dr. Smith sent a PET scan order.'

print("Loading AMR parser from custom path...")
stog = load_inference_model(custom_model_path)

# Parse sentence to AMR string
amr_strings = stog.parse_sents([test_sentence])
print("\n[AMR Output]")
print(amr_strings[0])

# Convert AMR string to Penman graph
print("\n[Penman Graph]")
g = penman.decode(amr_strings[0])
print(g)

# Convert to networkx graph for visualization
print("\n[Graph Visualization]")
G = nx.DiGraph()
for triple in g.triples:
    src, rel, tgt = triple
    G.add_node(src)
    G.add_node(tgt)
    G.add_edge(src, tgt, label=rel)

# Draw graph
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold')
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)
plt.title("AMR Graph of the Sentence")
plt.axis('off')
plt.tight_layout()
plt.show()
