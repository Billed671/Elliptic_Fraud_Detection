import numpy as np

from elliptic_fraud.metrics import binary_metrics, choose_f1_threshold, evaluate_by_horizon


def test_metrics_use_illicit_as_positive_class():
    y = np.array([0, 1, 0, 1])
    probabilities = np.array([0.1, 0.9, 0.2, 0.8])
    threshold = choose_f1_threshold(y, probabilities)
    result = binary_metrics(y, probabilities, threshold)
    assert result["pr_auc"] == 1.0
    assert result["roc_auc"] == 1.0
    assert result["f1"] == 1.0
    assert "accuracy" not in result


def test_horizon_metrics_only_include_known_labels():
    y = np.array([0, 1, 2, 0, 1])
    time = np.array([35, 36, 37, 40, 49])
    p = np.array([0.1, 0.9, 0.99, 0.2, 0.8])
    rows = evaluate_by_horizon(y, p, time, threshold=0.5)
    first = rows[0]
    assert first["n"] == 2
    assert first["positives"] == 1
