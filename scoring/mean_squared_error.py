"""Regression evaluation metrics."""

import numpy as np

from .utils import _as_target_matrix


def mean_squared_error(
    pred_y: np.ndarray,
    true_y: np.ndarray
):
    """Calculate the mean squared error between predictions and targets.

    The mean squared error is the average squared difference between the
    predicted and true target values. One-dimensional prediction arrays are
    reshaped to match the shape of the target array.

    Args:
        pred_y: Predicted target values. May have shape ``(num_samples,)`` or
            ``(num_samples, num_targets)``.
        true_y: Ground-truth target values with shape ``(num_samples,)`` or
            ``(num_samples, num_targets)``.

    Returns:
        The mean squared error averaged over all samples and targets.

    Raises:
        RuntimeError: If the predictions and ground-truth values contain
            different numbers of samples.
    """
    pred_y = _as_target_matrix(pred_y, name="pred_y")
    true_y = _as_target_matrix(true_y, name="true_y")

    if pred_y.shape != true_y.shape:
        raise ValueError(
            "pred_y and true_y must have matching shapes after normalization, "
            f"but received {pred_y.shape} and {true_y.shape}."
        )

    return np.mean((pred_y - true_y) ** 2)
