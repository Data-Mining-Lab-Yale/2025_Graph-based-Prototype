# text_to_graph/visualize.py
import matplotlib.pyplot as plt
import networkx as nx

# def visualize_graph(G, title="Graph", save_path=None):
#     pos = nx.spring_layout(G, seed=42)
#     labels = nx.get_node_attributes(G, "label")
#     edge_labels = nx.get_edge_attributes(G, "dep")

#     plt.figure(figsize=(10, 6))
#     nx.draw(G, pos, with_labels=True, labels=labels,
#             node_size=2500, node_color="#BFD7EA", font_size=10, arrows=True)
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
#     plt.title(title)

#     if save_path:
#         plt.savefig(save_path)
#         print(f"Saved to {save_path}")
#     else:
#         plt.show()

# def visualize_graph(G, title="Graph", layout="spring", font_size=10, node_color="#BFD7EA", output_path=None):
#     import matplotlib.pyplot as plt
#     import networkx as nx

#     if layout == "spring":
#         pos = nx.spring_layout(G, seed=42)
#     elif layout == "shell":
#         pos = nx.shell_layout(G)
#     else:
#         pos = nx.kamada_kawai_layout(G)

#     labels = nx.get_node_attributes(G, "text")
#     edge_labels = nx.get_edge_attributes(G, "type")

#     plt.figure(figsize=(12, 8))
#     nx.draw(G, pos, with_labels=True, labels=labels, node_size=2800,
#             node_color=node_color, font_size=font_size, edge_color="gray", arrows=True)
#     if edge_labels:
#         nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

#     plt.title(title)
#     plt.tight_layout()

#     if output_path:
#         plt.savefig(output_path)
#         plt.close()
#     else:
#         plt.show()


def visualize_graph(
    G,
    title="Graph",
    layout="spring",
    font_size=10,
    node_color="#BFD7EA",
    output_path=None
):
    if layout == "spring":
        pos = nx.spring_layout(G, seed=42)
    elif layout == "shell":
        pos = nx.shell_layout(G)
    else:
        pos = nx.kamada_kawai_layout(G)

    # Show node labels using 'label' or 'text' attribute
    labels = {
        n: d.get("label", d.get("text", str(n)))
        for n, d in G.nodes(data=True)
    }

    # Optional: show dependency edge labels if they exist
    edge_labels = nx.get_edge_attributes(G, "dep") or nx.get_edge_attributes(G, "type")

    plt.figure(figsize=(12, 8))
    nx.draw(
        G, pos,
        with_labels=True,
        labels=labels,
        node_size=2800,
        node_color=node_color,
        font_size=font_size,
        edge_color="gray",
        arrows=True
    )

    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title(title)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
