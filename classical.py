"""Random Forest and XGBoost baselines."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from elliptic_fraud.features import fit_training_scaler
from elliptic_fraud.metrics import choose_f1_threshold, evaluate_by_horizon
from elliptic_fraud.splits import TemporalSplit


def _make_estimator(name: str, y_train: np.ndarray, seed: int):
    positives = max(int(np.sum(y_train == 1)), 1)
    negatives = max(int(np.sum(y_train == 0)), 1)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=350,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=350,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            min_child_weight=2,
            scale_pos_weight=negatives / positives,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(f"Unknown classical estimator: {name}")


def run_classical_model(
    *,
    model_name: str,
    feature_name: str,
    features: np.ndarray,
    y: np.ndarray,
    time_step: np.ndarray,
    split: TemporalSplit,
    output_dir: Path,
    seed: int,
) -> tuple[list[dict], float, float]:
    """Validate on steps 30-34, refit on steps 1-34, evaluate steps 35-49."""
    early_x, early_scaler = fit_training_scaler(features, split.early_train)
    validation_model = _make_estimator(model_name, y[split.early_train], seed)
    validation_model.fit(early_x[split.early_train], (y[split.early_train] == 1).astype(int))
    val_prob = validation_model.predict_proba(early_x[split.validation])[:, 1]
    val_truth = (y[split.validation] == 1).astype(int)
    threshold = choose_f1_threshold(val_truth, val_prob)
    val_pr_auc = float(average_precision_score(val_truth, val_prob))

    final_x, final_scaler = fit_training_scaler(features, split.final_train)
    final_model = _make_estimator(model_name, y[split.final_train], seed)
    final_model.fit(final_x[split.final_train], (y[split.final_train] == 1).astype(int))
    test_prob = final_model.predict_proba(final_x)[:, 1]
    rows = evaluate_by_horizon(y, test_prob, time_step, threshold)

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": final_model,
        "scaler": final_scaler,
        "feature_name": feature_name,
        "threshold": threshold,
    }
    joblib.dump(artifact, output_dir / f"{model_name}_{feature_name}.joblib")
    return rows, val_pr_auc, threshold
