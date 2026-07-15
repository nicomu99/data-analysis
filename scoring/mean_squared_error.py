from typing import Any

import numpy as np
from numpy import floating


def mean_squared_error(
    pred_y: np.ndarray,
    true_y: np.ndarray
) -> floating[Any]:
    return np.mean((pred_y - true_y) ** 2)
