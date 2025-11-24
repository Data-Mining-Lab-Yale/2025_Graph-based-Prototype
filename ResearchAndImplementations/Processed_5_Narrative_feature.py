import json
import torch
from pathlib import Path
from torch_geometric.data import Data

# === Config ===
NARRATIVE_GRAPH_DIR = Path("filtered_narrative_ego_graphs")
AMR_FEATURE_PT = Path("processed_graph_features_amr.pt")
OUTPUT_PT = Path("processed_graph_features_narrative_ego.pt")

# === Load AMR dataset (.pt) and build lookup
amr_dataset = torch.load(AMR_FEATURE_PT, weights_only=False)
amr_dict = {data.id: data for data in amr_dataset}
print(f"✅ Loaded AMR dataset: {len(amr_dict)} entries")

dataset = []
skipped = []

for fpath in NARRATIVE_GRAPH_DIR.glob("*.json"):
    with open(fpath, "r", encoding="utf-8") as f:
        graph = json.load(f)

    center_id = graph.get("center_id")
    if center_id is None or center_id not in amr_dict:
        skipped.append((fpath.name, "Missing AMR for center_id"))
        continue

    # Get AMR mean feature for center
    center_feat = amr_dict[center_id].x.mean(dim=0)

    # Build node features with weighted AMR vectors
    node_id_map = {node["id"]: i for i, node in enumerate(graph["nodes"])}
    node_features = []

    for node in graph["nodes"]:
        sid = node["id"]
        if sid in amr_dict:
            base_feat = amr_dict[sid].x.mean(dim=0)
        else:
            base_feat = torch.zeros_like(center_feat)

        # Default: 1.0 for center, 0.0 for others unless edge is found
        weight = 1.0 if sid == center_id else 0.0
        for edge in graph.get("edges", []):
            if edge["source"] == center_id and edge["target"] == sid:
                weight = edge.get("weight", 1.0)
                break
        node_features.append(base_feat * weight)

    x = torch.stack(node_features)
    edge_index = torch.tensor(
        [[node_id_map[e["source"]], node_id_map[e["target"]]] for e in graph.get("edges", [])],
        dtype=torch.long
    ).t().contiguous()

    edge_attr = torch.tensor(
        [e.get("weight", 1.0) for e in graph.get("edges", [])],
        dtype=torch.float
    ).unsqueeze(1)

    # data = Data(
    #     x=x,
    #     edge_index=edge_index,
    #     edge_attr=edge_attr,
    #     y=graph.get("label", ""),
    #     id=center_id,
    #     text=graph.get("text", "")
    # )

    # Get the integer node index of the center node
    center_index = node_id_map[center_id]

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=graph.get("label", ""),
        id=center_id,  # keep string ID
        center_index=torch.tensor(center_index),  # add correct node index
        text=graph.get("text", "")
    )


    dataset.append(data)

# Save result
torch.save(dataset, OUTPUT_PT)
print(f"\n✅ Saved {len(dataset)} narrative GCN examples to: {OUTPUT_PT}")
if skipped:
    print(f"⚠️ Skipped {len(skipped)} graphs:")
    for fname, reason in skipped[:5]:
        print(f" - {fname}: {reason}")
