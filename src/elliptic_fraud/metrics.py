"""Metrics frozen by the proposal: PR-AUC first, F1 and ROC-AUC secondary."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score


def choose_f1_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Choose a decision threshold on validation data only."""
    if np.unique(y_true).size < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if thresholds.size == 0:
        return 0.5
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(thresholds[int(np.nanargmax(f1))])


def binary_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float | int]:
    """Compute metrics without accuracy; positive class is illicit (label 1)."""
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if y_true.size == 0:
        return {"n": 0, "positives": 0, "pr_auc": float("nan"), "roc_auc": float("nan"), "f1": float("nan")}
    positives = int(y_true.sum())
    pr_auc = float(average_precision_score(y_true, probabilities)) if positives else float("nan")
    roc_auc = (
        float(roc_auc_score(y_true, probabilities))
        if np.unique(y_true).size == 2
        else float("nan")
    )
    predicted = (probabilities >= threshold).astype(np.int64)
    return {
        "n": int(y_true.size),
        "positives": positives,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
    }


def evaluate_by_horizon(
    y: np.ndarray,
    probabilities: np.ndarray,
    time_step: np.ndarray,
    threshold: float,
) -> list[dict[str, float | int | str]]:
    """Evaluate known labels by forecast horizon after the time-step-34 cutoff."""
    specs = (
        ("1-5 steps ahead", 35, 39),
        ("6-10 steps ahead", 40, 44),
        ("11-15 steps ahead", 45, 49),
        ("all test steps", 35, 49),
    )
    rows: list[dict[str, float | int | str]] = []
    for name, start, end in specs:
        mask = (time_step >= start) & (time_step <= end) & (y != 2)
        values = binary_metrics((y[mask] == 1).astype(int), probabilities[mask], threshold)
        rows.append({"horizon": name, **values})
    return rows
