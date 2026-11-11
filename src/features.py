"""Leakage-conscious tabular and structural feature construction."""

from __future__ import annotations

import networkx as nx
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.data import EllipticGraph

STRUCTURAL_FEATURE_NAMES = ("pagerank", "degree_centrality", "clustering_coefficient")


def build_structural_features(graph: EllipticGraph) -> np.ndarray:
    """Compute PageRank, degree centrality, and clustering inside each time slice.

    No edge that connects different time steps contributes to these metrics.
    PageRank uses the directed payment-flow graph; degree centrality and
    clustering use its undirected projection.
    """
    n_nodes = graph.num_nodes
    result = np.zeros((n_nodes, 3), dtype=np.float32)
    src, dst = graph.edge_index
    same_step = graph.time_step[src] == graph.time_step[dst]
    src = src[same_step]
    dst = dst[same_step]

    for step in np.unique(graph.time_step):
        node_ids = np.flatnonzero(graph.time_step == step)
        if node_ids.size == 0:
            continue
        # Re-index to compact local integer IDs for efficient NetworkX work.
        local_id = {int(node_id): i for i, node_id in enumerate(node_ids)}
        graph_mask = graph.time_step[src] == step
        step_src = src[graph_mask]
        step_dst = dst[graph_mask]
        directed = nx.DiGraph()
        directed.add_nodes_from(range(len(node_ids)))
        directed.add_edges_from(
            (local_id[int(u)], local_id[int(v)])
            for u, v in zip(step_src, step_dst, strict=False)
        )
        undirected = directed.to_undirected()

        pagerank = nx.pagerank(directed, alpha=0.85, max_iter=100, tol=1e-6)
        degree = nx.degree_centrality(undirected)
        clustering = nx.clustering(undirected)
        for local_idx, global_idx in enumerate(node_ids):
            result[global_idx, 0] = pagerank[local_idx]
            result[global_idx, 1] = degree[local_idx]
            result[global_idx, 2] = clustering[local_idx]
        print(f"Computed structural features for time step {int(step)} ({len(node_ids):,} nodes).")

    return result


def build_feature_matrices(
    graph: EllipticGraph, structural: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw+time and raw+time+structural matrices."""
    if structural.shape != (graph.num_nodes, 3):
        raise ValueError(
            f"Expected structural feature shape {(graph.num_nodes, 3)}, got {structural.shape}."
        )
    time_feature = graph.time_step.astype(np.float32).reshape(-1, 1)
    raw_plus_time = np.concatenate([graph.x, time_feature], axis=1)
    engineered = np.concatenate([raw_plus_time, structural.astype(np.float32)], axis=1)
    return raw_plus_time, engineered


def fit_training_scaler(
    features: np.ndarray, train_indices: np.ndarray
) -> tuple[np.ndarray, StandardScaler]:
    """Fit scaling statistics on the provided training nodes only."""
    if train_indices.size == 0:
        raise ValueError("Cannot fit a scaler on an empty training split.")
    scaler = StandardScaler()
    scaler.fit(features[train_indices])
    transformed = scaler.transform(features).astype(np.float32, copy=False)
    return transformed, scaler
