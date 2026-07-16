"""Probabilistic evaluation metrics"""

import numpy as np

from .utils import _as_target_matrix


def neg_log_probability(
    mean_y: np.ndarray,
    var_y: np.ndarray,
    true_y: np.ndarray
):
    """Calculate the negative log likelihood of observed values.

    The negative log probability is the negative logarithm of the density
    assigned to observed values. If a model is capable of predicting a
    distribution rather than a single number, this function computes the
    negative log probability under the assumption that the predictions follow
    a normal distribution.

    Args:
        mean_y: Predicted target value means. May have shape ``(num_samples,)``
            or ``(num_samples, num_targets)``.
        var_y: Predicted target value variances. May have shape
            ``(num_samples,)`` or ``(num_samples, num_targets)``.
        true_y: Ground-truth target values with shape ``(num_samples,)`` or
            ``(num_samples, num_targets)``.

    Returns:
        The mean negative log probability averaged over all samples and
        targets.

    Raises:
        RuntimeError: If the predictions and ground-truth values contain
            different numbers of samples.
    """
    mean_y = _as_target_matrix(mean_y, name="mean_y")
    var_y = _as_target_matrix(var_y, name="var_y")
    true_y = _as_target_matrix(true_y, name="true_y")

    return -1 * np.mean(
        -0.5 * (
            np.log(2.0 * np.pi * var_y) +
            (true_y - mean_y) ** 2 / var_y
        )
    )
