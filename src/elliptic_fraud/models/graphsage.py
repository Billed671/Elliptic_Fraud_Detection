"""Two-layer GraphSAGE baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch import nn
from torch_geometric.nn import SAGEConv

from elliptic_fraud.metrics import choose_f1_threshold, evaluate_by_horizon
from elliptic_fraud.splits import TemporalSplit


class GraphSAGE(nn.Module):
    """Two-layer node classifier with a scalar illicit-class logit."""

    def __init__(self, in_channels: int, hidden_channels: int = 64, dropout: float = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index).squeeze(-1)


def _fit_graphsage(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    train_idx: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    epochs: int,
    validation_idx: np.ndarray | None = None,
) -> tuple[GraphSAGE, int, float | None]:
    torch.manual_seed(seed)
    model = GraphSAGE(x.shape[1]).to(device)
    x_device, edge_device, y_device = x.to(device), edge_index.to(device), y.to(device)
    train_tensor = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    train_labels = y_device[train_tensor]
    positives = torch.clamp(train_labels.sum(), min=1)
    negatives = torch.clamp((1 - train_labels).sum(), min=1)
    pos_weight = (negatives / positives).to(torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_score = -float("inf")
    best_epoch = epochs
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x_device, edge_device)
        loss = F.binary_cross_entropy_with_logits(
            logits[train_tensor], train_labels.float(), pos_weight=pos_weight
        )
        loss.backward()
        optimizer.step()

        if validation_idx is not None:
            model.eval()
            with torch.no_grad():
                val_tensor = torch.as_tensor(validation_idx, dtype=torch.long, device=device)
                probabilities = torch.sigmoid(model(x_device, edge_device)[val_tensor]).cpu().numpy()
            truth = (y[validation_idx] == 1).astype(int)
            score = float(average_precision_score(truth, probabilities))
            print(f"GraphSAGE epoch {epoch}/{epochs} - loss={loss.item():.4f} - val PR-AUC={score:.4f}")
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            print(f"GraphSAGE epoch {epoch}/{epochs} - loss={loss.item():.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, (None if validation_idx is None else best_score)


def run_graphsage(
    *,
    validation_x: np.ndarray,
    final_x: np.ndarray,
    edge_index: np.ndarray,
    y: np.ndarray,
    time_step: np.ndarray,
    split: TemporalSplit,
    output_dir: Path,
    device: torch.device,
    seed: int,
    epochs: int = 20,
) -> tuple[list[dict], float, float, int]:
    """Validate on steps 30-34, then refit on all known labels in steps 1-34."""
    validation_tensor = torch.as_tensor(validation_x, dtype=torch.float32)
    final_tensor = torch.as_tensor(final_x, dtype=torch.float32)
    edge_tensor = torch.as_tensor(edge_index, dtype=torch.long)
    y_tensor = torch.as_tensor((y == 1).astype(np.float32))

    validation_model, best_epoch, val_score = _fit_graphsage(
        validation_tensor,
        edge_tensor,
        y_tensor,
        split.early_train,
        device=device,
        seed=seed,
        epochs=epochs,
        validation_idx=split.validation,
    )
    if val_score is None:
        raise RuntimeError("GraphSAGE validation score was not produced.")
    validation_model.eval()
    with torch.no_grad():
        val_tensor = torch.as_tensor(split.validation, dtype=torch.long, device=device)
        val_logits = validation_model(validation_tensor.to(device), edge_tensor.to(device))
        val_prob = torch.sigmoid(val_logits[val_tensor]).cpu().numpy()
    threshold = choose_f1_threshold((y[split.validation] == 1).astype(int), val_prob)

    final_model, _, _ = _fit_graphsage(
        final_tensor,
        edge_tensor,
        y_tensor,
        split.final_train,
        device=device,
        seed=seed,
        epochs=best_epoch,
    )
    final_model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(final_model(final_tensor.to(device), edge_tensor.to(device)))
        test_prob = probabilities.cpu().numpy()
    rows = evaluate_by_horizon(y, test_prob, time_step, threshold)

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": final_model.state_dict(),
            "input_dim": final_x.shape[1],
            "hidden_dim": 64,
            "threshold": threshold,
            "epochs": best_epoch,
        },
        output_dir / "graphsage.pt",
    )
    return rows, val_score, threshold, best_epoch
