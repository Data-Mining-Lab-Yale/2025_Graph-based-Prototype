import networkx as nx
import matplotlib.pyplot as plt
from transformers import pipeline

# 1. Load model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# 2. Sample sentence and labels
sentence = "The patient took the medication before dinner."
labels = ["action", "agent", "time", "location", "recipient"]

# 3. Run classification
result = classifier(sentence, candidate_labels=labels)

# 4. Build graph
G = nx.DiGraph()
G.add_node("Sentence", label=sentence)

for label, score in zip(result["labels"], result["scores"]):
    G.add_node(label, score=score)
    G.add_edge("Sentence", label, weight=score)

# 5. Visualize
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G, seed=42)
edge_labels = { (u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True) }

nx.draw(G, pos, with_labels=True, node_size=3000, node_color="#A7C7E7", font_size=10)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title("Semantic Role Graph from Zero-Shot Classification")
plt.tight_layout()
plt.savefig("srl_graph_output.png")
plt.show()
