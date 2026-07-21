"""Negative ELBO Loss."""

import torch


def negative_elbo(
    mean_y: torch.Tensor,
    log_var_y: torch.Tensor,
    true_y: torch.Tensor,
    kl_divergence: torch.Tensor
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
        kl_divergence: KL-divergence of the model weights.

    Returns:
        The mean Gaussian negative log-likelihood over all predictions.
    """
    num_samples = mean_y.shape[0]

    var_y = torch.exp(log_var_y)
    likelihood_loss = 0.5 * (
            log_var_y + (true_y - mean_y).square() / var_y
    ).mean()
    return kl_divergence / num_samples + likelihood_loss
