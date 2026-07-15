"""Probabilistic evaluation metrics"""

import numpy as np

from .utils import _as_target_matrix


def log_probability(
    mean_y: np.ndarray,
    var_y: np.ndarray,
    true_y: np.ndarray
):
    mean_y = _as_target_matrix(mean_y, name="mean_y")
    var_y = _as_target_matrix(var_y, name="var_y")
    true_y = _as_target_matrix(true_y, name="true_y")

    return np.mean(
        -0.5 * (
            np.log(2.0 * np.pi * var_y) +
            (true_y - mean_y) ** 2 / var_y
        )
    )
