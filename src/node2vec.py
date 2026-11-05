"""Node2Vec structural embedding baseline with a logistic-regression head."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from elliptic_fraud.metrics import choose_f1_threshold, evaluate_by_horizon
from elliptic_fraud.splits import TemporalSplit


def learn_embeddings(
    edge_index: np.ndarray,
    num_nodes: int,
    *,
    device: torch.device,
    seed: int,
    embedding_dim: int = 64,
    epochs: int = 5,
) -> np.ndarray:
    """Learn unsupervised Node2Vec embeddings over the observed full graph.

    This is a transductive topology-only baseline: test-node connectivity is
    visible during embedding learning, but no test labels are used. See README.
    """
    try:
        from torch_geometric.nn import Node2Vec
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Node2Vec requires torch-geometric.") from exc

    torch.manual_seed(seed)
    edges = torch.as_tensor(edge_index, dtype=torch.long)
    model = Node2Vec(
        edges,
        embedding_dim=embedding_dim,
        walk_length=20,
        context_size=10,
        walks_per_node=5,
        num_negative_samples=1,
        p=1.0,
        q=1.0,
        sparse=True,
        num_nodes=num_nodes,
    ).to(device)
    try:
        loader = model.loader(batch_size=256, shuffle=True, num_workers=0)
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
        for epoch in range(1, epochs + 1):
            model.train()
            losses = []
            for positive_rw, negative_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(positive_rw.to(device), negative_rw.to(device))
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            if not losses:
                raise RuntimeError("Node2Vec generated no random walks.")
            print(f"Node2Vec epoch {epoch}/{epochs} - loss={np.mean(losses):.4f}")
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "Node2Vec random-walk sampling needs a compatible PyG extension "
            "(pyg-lib or torch-cluster). Install the extension matching your "
            "PyTorch/CUDA build; see https://pytorch-geometric.readthedocs.io/."
        ) from exc

    model.eval()
    with torch.no_grad():
        return model.embedding.weight.detach().cpu().numpy().astype(np.float32)


def run_node2vec(
    *,
    edge_index: np.ndarray,
    num_nodes: int,
    y: np.ndarray,
    time_step: np.ndarray,
    split: TemporalSplit,
    output_dir: Path,
    device: torch.device,
    seed: int,
) -> tuple[list[dict], float, float]:
    embeddings = learn_embeddings(edge_index, num_nodes, device=device, seed=seed)
    validation_scaler = StandardScaler().fit(embeddings[split.early_train])
    validation_embeddings = validation_scaler.transform(embeddings).astype(np.float32)
    validation_model = LogisticRegression(
        max_iter=1000, class_weight="balanced", C=1.0, random_state=seed
    )
    validation_model.fit(
        validation_embeddings[split.early_train], (y[split.early_train] == 1).astype(int)
    )
    val_prob = validation_model.predict_proba(validation_embeddings[split.validation])[:, 1]
    val_truth = (y[split.validation] == 1).astype(int)
    threshold = choose_f1_threshold(val_truth, val_prob)
    val_pr_auc = float(average_precision_score(val_truth, val_prob))

    final_scaler = StandardScaler().fit(embeddings[split.final_train])
    final_embeddings = final_scaler.transform(embeddings).astype(np.float32)
    final_model = LogisticRegression(
        max_iter=1000, class_weight="balanced", C=1.0, random_state=seed
    )
    final_model.fit(final_embeddings[split.final_train], (y[split.final_train] == 1).astype(int))
    test_prob = final_model.predict_proba(final_embeddings)[:, 1]
    rows = evaluate_by_horizon(y, test_prob, time_step, threshold)

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final_model,
            "scaler": final_scaler,
            "threshold": threshold,
            "embedding_dim": embeddings.shape[1],
        },
        output_dir / "node2vec_logistic.joblib",
    )
    np.save(output_dir / "node2vec_embeddings.npy", embeddings)
    return rows, val_pr_auc, threshold
