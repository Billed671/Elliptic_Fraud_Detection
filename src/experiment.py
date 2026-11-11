"""End-to-end experiment orchestration and artifact writing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_elliptic
from src.features import build_feature_matrices, build_structural_features, fit_training_scaler
from src.models.classical import run_classical_model
from src.splits import make_temporal_split


def run_experiment(
    *,
    data_dir: str | Path = "data",
    output_dir: str | Path = "outputs",
    models: tuple[str, ...] = ("classical", "graphsage", "node2vec"),
    seed: int = 42,
    graphsage_epochs: int = 20,
) -> Path:
    """Run requested baselines and return the metrics CSV path."""
    output_dir = Path(output_dir)
    model_dir = output_dir / "models"
    report_dir = output_dir / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    graph = load_elliptic(data_dir)
    split = make_temporal_split(graph.y, graph.time_step)
    print(
        "Temporal split | early train (1-29): {:,} | validation (30-34): {:,} | "
        "final train (1-34): {:,} | test (35-49): {:,}".format(
            len(split.early_train), len(split.validation), len(split.final_train), len(split.test)
        )
    )

    raw_plus_time = engineered = None
    if "classical" in models:
        structural = build_structural_features(graph)
        raw_plus_time, engineered = build_feature_matrices(graph, structural)
    rows: list[dict] = []
    validation_scores: dict[str, float] = {}
    thresholds: dict[str, float] = {}

    if "classical" in models:
        assert raw_plus_time is not None and engineered is not None
        for estimator in ("random_forest", "xgboost"):
            for feature_name, features in (
                ("raw", raw_plus_time),
                ("engineered", engineered),
            ):
                name = f"{estimator}_{feature_name}"
                print(f"\n=== {name} ===")
                model_rows, val_score, threshold = run_classical_model(
                    model_name=estimator,
                    feature_name=feature_name,
                    features=features,
                    y=graph.y,
                    time_step=graph.time_step,
                    split=split,
                    output_dir=model_dir,
                    seed=seed,
                )
                validation_scores[name] = val_score
                thresholds[name] = threshold
                rows.extend({"model": name, **row} for row in model_rows)
                print(f"Validation PR-AUC: {val_score:.4f} | F1 threshold: {threshold:.4f}")

    if "node2vec" in models:
        print("\n=== node2vec_logistic ===")
        from src.models.node2vec import run_node2vec   # lazy import
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Used device for Node2Vec: {device}")

        model_rows, val_score, threshold = run_node2vec(
            edge_index=graph.edge_index,
            num_nodes=graph.num_nodes,
            y=graph.y,
            time_step=graph.time_step,
            split=split,
            output_dir=model_dir,
            device=device,
            seed=seed,
        )
        validation_scores["node2vec_logistic"] = val_score
        thresholds["node2vec_logistic"] = threshold
        rows.extend({"model": "node2vec_logistic", **row} for row in model_rows)
        print(f"Validation PR-AUC: {val_score:.4f} | F1 threshold: {threshold:.4f}")

    if "graphsage" in models:
        print("\n=== graphsage ===")
        from src.models.graphsage import run_graphsage   # lazy import
        import torch
        # Proposal baseline: original 165 node features + graph structure.
        # Scaling uses only labeled early-train nodes for the validation run.
        early_x, _ = fit_training_scaler(graph.x, split.early_train)
        final_x, _ = fit_training_scaler(graph.x, split.final_train)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if str(device) == "cpu":
            if not torch.backends.mps.is_available():
                if not torch.backends.mps.is_built():
                    print("MPS not available because the current PyTorch install was not "
                          "built with MPS enabled.")
                else:
                    print("MPS not available because the current macOS version is not 15.0+ "
                          "and/or you do not have an MPS-enabled device on this machine.")

            else:
                device = torch.device("mps")

        print(f"Used device for GraphSAGE: {device}")

        model_rows, val_score, threshold, best_epoch = run_graphsage(
            validation_x=early_x,
            final_x=final_x,
            edge_index=graph.edge_index,
            y=graph.y,
            time_step=graph.time_step,
            split=split,
            output_dir=model_dir,
            device=device,
            seed=seed,
            epochs=graphsage_epochs,
        )
        validation_scores["graphsage"] = val_score
        thresholds["graphsage"] = threshold
        rows.extend({"model": "graphsage", **row} for row in model_rows)
        print(f"Validation PR-AUC: {val_score:.4f} | best epoch: {best_epoch}")

    ml_scores = [score for name, score in validation_scores.items() if name.startswith(("random_forest", "xgboost"))]
    fallback_triggered = bool(
        "graphsage" in validation_scores and ml_scores and validation_scores["graphsage"] < max(ml_scores)
    )
    summary = {
        "seed": seed,
        "split": {
            "early_train_steps": [1, 29],
            "validation_steps": [30, 34],
            "final_train_steps": [1, 34],
            "test_steps": [35, 49],
            "horizons": {"1-5": [35, 39], "6-10": [40, 44], "11-15": [45, 49]},
        },
        "validation_pr_auc": validation_scores,
        "f1_thresholds_from_validation": thresholds,
        "graphsage_fallback_triggered": fallback_triggered,
        "fallback_rule": "Trigger when GraphSAGE validation PR-AUC is below the best classical ML validation PR-AUC.",
        "class_counts": {
            "illicit": int(np.sum(graph.y == 1)),
            "licit": int(np.sum(graph.y == 0)),
            "unknown": int(np.sum(graph.y == 2)),
        },
    }

    metrics_csv = report_dir / "metrics_by_horizon.csv"
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(metrics_csv, index=False)
    (report_dir / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\n=== Test results (PR-AUC is primary) ===")
    if not metrics_df.empty:
        print(metrics_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved reports to: {report_dir}")
    if fallback_triggered:
        print("FALLBACK: GraphSAGE validation PR-AUC is below the best classical ML baseline.")
    return metrics_csv
