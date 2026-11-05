"""Dataset loading and lightweight validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EllipticGraph:
    """Numpy-backed view of the PyG Elliptic graph.

    Label convention used by PyG: 0 = licit, 1 = illicit, 2 = unknown.
    The PyG dataset stores the 165 numeric columns in ``data.x`` but does not
    expose the time-step column on the Data object, so it is recovered from
    the original feature CSV and aligned by row order.
    """

    x: np.ndarray
    edge_index: np.ndarray
    y: np.ndarray
    time_step: np.ndarray

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def labeled_mask(self) -> np.ndarray:
        return self.y != 2


def load_elliptic(root: str | Path = "data") -> EllipticGraph:
    """Download/load ``EllipticBitcoinDataset`` and recover time-step values."""
    try:
        import pandas as pd
        import torch_geometric
        from torch_geometric.datasets import EllipticBitcoinDataset
    except ImportError as exc:  # pragma: no cover - depends on optional runtime setup
        raise RuntimeError(
            "PyTorch Geometric and pandas are required. Install project dependencies with "
            "`pip install -e .` and follow the PyG installation guide if needed."
        ) from exc

    dataset = EllipticBitcoinDataset(root=str(root))
    data = dataset[0]
    feature_path = Path(dataset.raw_paths[0])
    if not feature_path.exists():
        raise FileNotFoundError(
            f"Could not find the raw feature file at {feature_path}; "
            "the PyG dataset download may have failed."
        )

    # The first two raw columns are transaction ID and time_step; x contains
    # the remaining 165 numeric features.
    time_step = pd.read_csv(feature_path, header=None, usecols=[1]).iloc[:, 0]
    time_step = time_step.to_numpy(dtype=np.int64)

    x = data.x.detach().cpu().numpy().astype(np.float32, copy=False)
    edge_index = data.edge_index.detach().cpu().numpy().astype(np.int64, copy=False)
    y = data.y.detach().cpu().numpy().astype(np.int64, copy=False)

    if x.ndim != 2 or x.shape[1] != 165:
        raise ValueError(f"Expected 165 node features, received shape {x.shape}.")
    if len(time_step) != len(x) or len(y) != len(x):
        raise ValueError(
            "Feature, label, and time-step rows do not align: "
            f"x={len(x)}, y={len(y)}, time_step={len(time_step)}."
        )
    if edge_index.shape[0] != 2:
        raise ValueError(f"Expected edge_index with shape [2, num_edges], got {edge_index.shape}.")
    if not np.isin(y, [0, 1, 2]).all():
        raise ValueError("Unexpected label values. Expected 0=licit, 1=illicit, 2=unknown.")

    graph = EllipticGraph(x=x, edge_index=edge_index, y=y, time_step=time_step)
    print(
        f"Loaded {graph.num_nodes:,} nodes, {graph.num_edges:,} directed edges, "
        f"{graph.x.shape[1]} numeric features, "
        f"{len(np.unique(graph.time_step))} time steps (PyG {torch_geometric.__version__})."
    )
    labeled = graph.y[graph.labeled_mask]
    illicit = int(np.sum(labeled == 1))
    licit = int(np.sum(labeled == 0))
    total = illicit + licit
    if total:
        print(
            f"Known labels: {total:,} | illicit: {illicit:,} ({illicit / total:.2%}) | "
            f"licit: {licit:,} ({licit / total:.2%}) | unknown excluded."
        )
    return graph
