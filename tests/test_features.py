import numpy as np

from elliptic_fraud.data import EllipticGraph
from elliptic_fraud.features import build_feature_matrices, build_structural_features, fit_training_scaler


def test_structural_features_are_computed_within_time_steps():
    graph = EllipticGraph(
        x=np.zeros((4, 2), dtype=np.float32),
        edge_index=np.array([[0, 1, 2], [1, 0, 3]], dtype=np.int64),
        y=np.array([0, 1, 0, 1]),
        time_step=np.array([1, 1, 2, 2]),
    )
    structural = build_structural_features(graph)
    assert structural.shape == (4, 3)
    assert np.isfinite(structural).all()
    assert structural[0, 1] == 1.0
    assert structural[2, 1] == 1.0


def test_feature_matrices_have_expected_columns_and_scaler_fits_train_only():
    graph = EllipticGraph(
        x=np.array([[0.0, 1.0], [2.0, 3.0], [100.0, 100.0]], dtype=np.float32),
        edge_index=np.empty((2, 0), dtype=np.int64),
        y=np.array([0, 1, 0]),
        time_step=np.array([1, 2, 3]),
    )
    structural = np.zeros((3, 3), dtype=np.float32)
    raw, engineered = build_feature_matrices(graph, structural)
    assert raw.shape == (3, 3)
    assert engineered.shape == (3, 6)
    _, scaler = fit_training_scaler(raw, np.array([0, 1]))
    assert np.allclose(scaler.mean_, np.array([1.0, 2.0, 1.5]))
