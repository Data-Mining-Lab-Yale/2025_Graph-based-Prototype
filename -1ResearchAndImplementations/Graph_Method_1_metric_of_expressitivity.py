

#!/usr/bin/env python3
"""
Compute graph complexity and expressivity metrics for a folder of JSON graphs.

Changes in this version:
- Accept both "edges" and "links" in the input JSON.
- Write results as JSON: per-graph (graph_metrics.json) and summary (graph_metrics_summary.json).
- Keep CSV as optional for convenience.

Inputs:
- Directory with files like <subsentence_id>.json
- Each JSON may contain:
    {
      "nodes": [{"id": "...", "label": "...", "type": "...", ...}, ...],
      "edges": [{"source": "...", "target": "...", "label": "...", "type": "...", ...}, ...]
      // or "links" instead of "edges"
    }

Outputs:
- graph_metrics.json (list of per-graph dicts)
- graph_metrics_summary.json (dataset-level summary)
- graph_metrics.csv (optional convenience)

Dependencies:
    pip install networkx numpy pandas
"""

import os
import json
import math
import glob
import numpy as np
import pandas as pd
import networkx as nx

# FOLDER = "dependency_graphs"
# FOLDER = "srl_graphs_weighted"
# FOLDER = "srl_graphs_predicate"
# FOLDER = "srl_graphs_anchored"
# FOLDER = "amr_graphs"
FOLDER = "narrative_ego_graphs"

# ===================== USER CONFIG =====================
INPUT_DIR = f"outputs/{FOLDER}/subsentence_subcode/json"  # <-- change this
OUTPUT_PER_GRAPH_JSON = f"graph_metrics_{FOLDER}.json"
OUTPUT_SUMMARY_JSON = f"graph_metrics_summary_{FOLDER}.json"
OUTPUT_CSV = f"graph_metrics_{FOLDER}.csv"  # optional
# ======================================================

# WL hashing (NetworkX >= 2.5)
try:
    from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash as wl_hash
except Exception:
    wl_hash = None  # fallback below


def read_graph_from_json(path):
    """Load a graph JSON and build a NetworkX MultiDiGraph with node/edge attributes."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = data.get("nodes", [])
    # Accept either "edges" or "links"
    edges = data.get("edges", [])
    if not edges:
        edges = data.get("links", [])

    G = nx.MultiDiGraph()
    # Add nodes
    for n in nodes:
        nid = n.get("id")
        if nid is None:
            nid = f"node_{len(G)}"
        attrs = {k: v for k, v in n.items() if k != "id"}
        G.add_node(nid, **attrs)

    # Add edges
    for e in edges:
        src = e.get("source", e.get("from"))
        tgt = e.get("target", e.get("to"))
        if src is None or tgt is None:
            continue
        attrs = {k: v for k, v in e.items() if k not in ("source", "from", "target", "to")}
        G.add_edge(src, tgt, **attrs)

    return G


def to_simple_undirected(G):
    """Convert MultiDiGraph to a simple undirected graph."""
    if G.number_of_nodes() == 0:
        return nx.Graph()
    UG = nx.Graph()
    for n, attrs in G.nodes(data=True):
        UG.add_node(n, **attrs)
    for u, v, edata in G.edges(data=True):
        if UG.has_edge(u, v):
            for key, val in edata.items():
                existing = UG[u][v].get(key)
                if existing is None:
                    UG[u][v][key] = val
                else:
                    if isinstance(existing, set):
                        existing.add(val)
                    else:
                        UG[u][v][key] = {existing, val}
        else:
            UG.add_edge(u, v, **edata)
    return UG


def entropy(values, base=2):
    if not values:
        return 0.0
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    probs = np.array(list(counts.values()), dtype=float)
    probs = probs / probs.sum()
    logs = np.log(probs) / np.log(base)
    return float(-np.sum(probs * logs))


def degree_entropy(Gu):
    degs = [d for _, d in Gu.degree()]
    return entropy(degs, base=2)


def label_entropy(G, which="node"):
    labels = []
    if which == "node":
        for _, a in G.nodes(data=True):
            lab = a.get("label") or a.get("type")
            if lab is not None:
                labels.append(lab)
    else:
        for _, _, a in G.edges(data=True):
            lab = a.get("label") or a.get("type")
            if lab is not None:
                labels.append(lab)
    return entropy(labels, base=2)


def graph_energy(Gu):
    if Gu.number_of_nodes() == 0:
        return 0.0
    A = nx.to_numpy_array(Gu, dtype=float)
    eigvals = np.linalg.eigvalsh(A)
    return float(np.sum(np.abs(eigvals)))


def spectral_diversity(Gu):
    n = Gu.number_of_nodes()
    if n == 0:
        return 0, 0.0
    L = nx.laplacian_matrix(Gu).toarray()
    eigvals = np.linalg.eigvalsh(L)
    uniq = np.unique(np.round(eigvals, 8))
    total = np.sum(eigvals)
    if total <= 0:
        spec_entropy = 0.0
    else:
        p = eigvals / total
        p = p[p > 0]
        spec_entropy = float(-np.sum(p * np.log2(p)))
    return int(len(uniq)), spec_entropy


def graph_diameter_and_aspl(Gu):
    if Gu.number_of_nodes() <= 1:
        return 0, 0.0
    comps = [Gu.subgraph(c).copy() for c in nx.connected_components(Gu)]
    largest = max(comps, key=lambda H: H.number_of_nodes())
    if largest.number_of_nodes() <= 1:
        return 0, 0.0
    try:
        dia = nx.diameter(largest)
    except Exception:
        dia = 0
    try:
        aspl = nx.average_shortest_path_length(largest)
    except Exception:
        aspl = 0.0
    return int(dia), float(aspl)


def wl_graph_hash(G):
    if wl_hash is None:
        degs = sorted([d for _, d in G.degree()])
        return "degseq_" + str(hash(tuple(degs)))
    H = G.copy()
    any_label = False
    for node, attrs in list(H.nodes(data=True)):
        lab = attrs.get("label") or attrs.get("type")
        if lab is not None:
            any_label = True
            H.nodes[node]["label"] = str(lab)
    if any_label:
        return wl_hash(H, node_attr="label")
    else:
        return wl_hash(H)


def compute_metrics_for_graph(G_path):
    G = read_graph_from_json(G_path)
    Gu = to_simple_undirected(G)

    n = Gu.number_of_nodes()
    m = Gu.number_of_edges()
    density = nx.density(Gu) if n > 1 else 0.0
    avg_degree = (2.0 * m / n) if n > 0 else 0.0

    dia, aspl = graph_diameter_and_aspl(Gu)
    clustering = nx.average_clustering(Gu) if n > 1 else 0.0

    deg_H = degree_entropy(Gu)
    node_label_H = label_entropy(G, which="node")
    edge_label_H = label_entropy(G, which="edge")

    num_unique_eigs, spec_entropy = spectral_diversity(Gu)
    energy = graph_energy(Gu)

    H = Gu.copy()
    for node, attrs in list(H.nodes(data=True)):
        lab = attrs.get("label") or attrs.get("type")
        if lab is not None:
            H.nodes[node]["label"] = str(lab)
    wl = wl_graph_hash(H)

    return {
        "file": os.path.basename(G_path),
        "n_nodes": n,
        "n_edges": m,
        "density": density,
        "avg_degree": avg_degree,
        "diameter_lcc": dia,
        "aspl_lcc": aspl,
        "avg_clustering": clustering,
        "degree_entropy": deg_H,
        "node_label_entropy": node_label_H,
        "edge_label_entropy": edge_label_H,
        "lap_num_unique_eigs": num_unique_eigs,
        "lap_spectral_entropy": spec_entropy,
        "adj_graph_energy": energy,
        "wl_hash": wl,
    }


def summarize_dataframe(df):
    def mean_std(col):
        return {
            "mean": float(df[col].mean()),
            "std": float(df[col].std(ddof=1)) if len(df) > 1 else 0.0
        }

    summary = {
        "num_graphs": int(len(df)),
        "metrics": {}
    }
    core_cols = [
        "n_nodes", "n_edges", "density", "avg_degree",
        "diameter_lcc", "aspl_lcc", "avg_clustering",
        "degree_entropy", "node_label_entropy", "edge_label_entropy",
        "lap_num_unique_eigs", "lap_spectral_entropy", "adj_graph_energy"
    ]
    for c in core_cols:
        if c in df.columns:
            summary["metrics"][c] = mean_std(c)

    if "wl_hash" in df.columns:
        unique_hashes = int(df["wl_hash"].nunique())
        summary["wl_distinguishability"] = {
            "unique": unique_hashes,
            "total": int(len(df)),
            "fraction_unique": float(unique_hashes / max(1, len(df)))
        }
    return summary


def main():
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.json")))
    if not files:
        raise SystemExit(f"No JSON files found in: {INPUT_DIR}")

    rows = []
    for p in files:
        try:
            rows.append(compute_metrics_for_graph(p))
        except Exception as e:
            rows.append({"file": os.path.basename(p), "error": str(e)})

    # Save per-graph as JSON and CSV
    with open(OUTPUT_PER_GRAPH_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    # Summary JSON (exclude errored rows)
    df_ok = df[df.get("error").isna()] if "error" in df.columns else df
    summary = summarize_dataframe(df_ok)

    with open(OUTPUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote per-graph JSON to: {OUTPUT_PER_GRAPH_JSON}")
    print(f"[OK] Wrote summary JSON to: {OUTPUT_SUMMARY_JSON}")
    print(f"[OK] Wrote CSV to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
