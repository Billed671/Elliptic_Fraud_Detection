import numpy as np

from elliptic_fraud.splits import make_temporal_split


def test_temporal_split_excludes_unknown_and_respects_time_boundaries():
    time = np.array([1, 29, 30, 34, 35, 39, 40, 44, 45, 49, 49])
    y = np.array([0, 1, 0, 2, 1, 0, 1, 0, 1, 0, 2])
    split = make_temporal_split(y, time)

    assert set(time[split.early_train]) == {1, 29}
    assert set(time[split.validation]) == {30}
    assert set(time[split.final_train]) == {1, 29, 30, 35, 39, 40, 44, 45, 49} - {35, 39, 40, 44, 45, 49}
    assert set(time[split.test]) == {35, 39, 40, 44, 45, 49}
    assert not np.any(y[split.early_train] == 2)
    assert not np.any(y[split.validation] == 2)
    assert not np.any(y[split.test] == 2)
    assert np.array_equal(np.flatnonzero(split.horizon_masks["horizon_1_5"]), np.array([4, 5]))


def test_final_train_covers_steps_one_through_34():
    time = np.arange(1, 50)
    y = np.zeros(49, dtype=int)
    split = make_temporal_split(y, time)
    assert time[split.final_train].max() == 34
    assert time[split.test].min() == 35
