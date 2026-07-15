from typing import Any

import numpy as np
from numpy import floating


def mean_squared_error(
    pred_y: np.ndarray,
    true_y: np.ndarray
) -> floating[Any]:
    num_samples, num_targets = true_y.shape[0], true_y.shape[1]

    if pred_y.shape[0] != num_samples:
        raise RuntimeError("Shape mismatch between predictions and ground truth.")

    if pred_y.ndim == 1:
        pred_y = pred_y.reshape((num_samples, num_targets))

    return np.mean((pred_y - true_y) ** 2)
