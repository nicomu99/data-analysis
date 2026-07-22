"""Stochastic Gradient Langevin Dynamics Loss."""

from typing import Literal

import torch


def sgdl_loss(
    y_mean: torch.Tensor,
    y_log_var: torch.Tensor,
    y_true: torch.Tensor,
    neg_log_prior: torch.Tensor | Literal[0],
    num_samples: int
) -> torch.Tensor:
    """Computes the stochastic gradient langevin dynamics loss.

    The loss consists of the negative log prior and the negative log likelihood.
    The model is assumed to predict both the conditional mean and the
    logarithm of the conditional variance for each target value. Predicting
    the log-variance ensures that the corresponding variance is positive.

    The constant term ``0.5 * log(2 * pi)`` is omitted because it does not
    affect optimization.

    Args:
        y_mean: Predicted target means.
        y_log_var: Predicted logarithms of the target variances. Must have a
            shape compatible with ``mean_y`` and ``true_y``.
        y_true: Ground-truth target values.
        neg_log_prior: Negative log prior of the model weights.
        num_samples: Number of samples in the dataset.

    Returns:
        The mean Gaussian negative log-likelihood over all predictions.
    """
    y_log_var = torch.clamp(y_log_var, min=-10.0, max=10.0)
    negative_log_likelihood = 0.5 * (
        y_log_var + (y_true - y_mean).square() * torch.exp(-y_log_var)
    ).mean() * num_samples

    return neg_log_prior + negative_log_likelihood
