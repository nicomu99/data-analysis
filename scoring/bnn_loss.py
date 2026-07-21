"""Loss Function for BNNs with Laplace Approximation."""

import torch


def bnn_loss(
    mean_y: torch.Tensor,
    log_var_y: torch.Tensor,
    true_y: torch.Tensor
):
    """Calculate the Gaussian negative log-likelihood loss.

    The model is assumed to predict both the conditional mean and the
    logarithm of the conditional variance for each target value. Predicting
    the log-variance ensures that the corresponding variance is positive.

    The constant term ``0.5 * log(2 * pi)`` is omitted because it does not
    affect optimization.

    Args:
        mean_y: Predicted target means.
        log_var_y: Predicted logarithms of the target variances. Must have a
            shape compatible with ``mean_y`` and ``true_y``.
        true_y: Ground-truth target values.

    Returns:
        The mean Gaussian negative log-likelihood over all predictions.
    """
    log_var_y = torch.clamp(log_var_y, min=-10.0, max=10.0)
    loss = log_var_y + (true_y - mean_y).square() / torch.exp(log_var_y)
    return 0.5 * loss.mean()
