# Text2Graph_3_amr_example.py
import amrlib
import penman
import networkx as nx
import matplotlib.pyplot as plt

# 1. Load model (after placing .tar.gz extracted under amrlib/data/model_stog)
print("🔄 Loading AMR parser model...")
stog = amrlib.load_stog_model()

# 2. Parse
sent = "Dr. Smith sent a PET scan order."
graphs = stog.parse_sents([sent])
print("✅ AMR parse PENMAN style:\n", graphs[0])

# 3. Convert to JSON-like and visualize
amr = penman.decode(graphs[0])
nodes, edges = amr.nodes, amr.edges()
json_graph = {
    "nodes": [{ "id": n.node_id, "label": n.target } for n in nodes],
    "edges": [{ "source": e.source, "target": e.target, "label": e.role } for e in edges]
}
print("✅ JSON:", json_graph)

# 4. Visualize with networkx
G = nx.DiGraph()
for n in json_graph["nodes"]:
    G.add_node(n["id"], label=n["label"])
for e in json_graph["edges"]:
    G.add_edge(e["source"], e["target"], label=e["label"])
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, labels={n["id"]:n["label"] for n in json_graph["nodes"]})
nx.draw_networkx_edge_labels(G, pos, edge_labels={(e["source"], e["target"]):e["label"] for e in json_graph["edges"]})
plt.show()
