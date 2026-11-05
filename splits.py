"""Explicit temporal splits. Unknown labels are never used for supervision or scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TemporalSplit:
    early_train: np.ndarray
    validation: np.ndarray
    final_train: np.ndarray
    test: np.ndarray
    horizon_masks: dict[str, np.ndarray]


def make_temporal_split(
    y: np.ndarray,
    time_step: np.ndarray,
    *,
    validation_start: int = 30,
    train_end: int = 34,
    test_start: int = 35,
    test_end: int = 49,
) -> TemporalSplit:
    """Create the proposal's 1-34 -> 35-49 split with a 30-34 validation window.

    Hyperparameter/threshold selection uses steps 1-29 vs. 30-34. Final models
    are refit on all known labels from steps 1-34 before test evaluation.
    """
    if y.shape != time_step.shape:
        raise ValueError("y and time_step must have the same shape.")
    known = y != 2
    early_train = np.flatnonzero(known & (time_step < validation_start))
    validation = np.flatnonzero(
        known & (time_step >= validation_start) & (time_step <= train_end)
    )
    final_train = np.flatnonzero(known & (time_step <= train_end))
    test = np.flatnonzero(
        known & (time_step >= test_start) & (time_step <= test_end)
    )
    masks = {
        "horizon_1_5": (time_step >= 35) & (time_step <= 39) & known,
        "horizon_6_10": (time_step >= 40) & (time_step <= 44) & known,
        "horizon_11_15": (time_step >= 45) & (time_step <= 49) & known,
        "all_test": (time_step >= test_start) & (time_step <= test_end) & known,
    }
    if min(len(early_train), len(validation), len(final_train), len(test)) == 0:
        raise ValueError("A temporal split is empty; inspect time-step and label values.")
    return TemporalSplit(early_train, validation, final_train, test, masks)
