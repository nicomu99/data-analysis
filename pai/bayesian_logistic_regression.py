"""Variational inference for Bayesian logistic regression."""

import torch
from torch.optim import SGD
import torch.nn.functional as f


def vi_loss(
    logits: torch.Tensor,
    y_true: torch.Tensor,
    mean: torch.Tensor,
    covariance: torch.Tensor,
    num_samples: int
):
    """Calculate the variational inference loss.

    The loss is the negative evidence lower bound for Bayesian logistic
    regression. It consists of the expected negative log-likelihood and the
    Kullback-Leibler divergence between the Gaussian variational distribution
    and a standard multivariate normal prior.

    The expected negative log-likelihood is approximated using Monte Carlo
    samples from the variational distribution.

    Args:
        logits: Logistic regression logits for each observation and Monte
            Carlo sample, with shape ``(num_observations, num_mc_samples)``.
        y_true: Binary target labels encoded as ``-1`` and ``1``, with shape
            ``(num_observations,)``.
        mean: Mean vector of the Gaussian variational distribution, with shape
            ``(num_parameters,)``.
        covariance: Covariance matrix of the Gaussian variational
            distribution, with shape
            ``(num_parameters, num_parameters)``.
        num_samples: Number of observations in the dataset. This is used to
            scale the KL-divergence term when the likelihood is averaged over
            observations.

    Returns:
        The scalar negative evidence lower bound.
    """
    likelihood_loss = torch.mean(
        torch.log1p(torch.exp(-y_true[:, None] * logits))
    )
    kl_loss = (
        covariance.trace() + mean.square().sum() - mean.numel() - torch.log(torch.det(covariance))
    ) / 2

    return likelihood_loss + kl_loss / num_samples


def bayesian_logistic_vi(
    x: torch.Tensor,
    y: torch.Tensor
):
    """Fit Bayesian logistic regression using variational inference.

    The posterior distribution over the regression coefficients is
    approximated by a full-covariance multivariate Gaussian distribution.

    The covariance matrix is parameterized through a lower-triangular Cholesky
    factor. Its diagonal entries are transformed using the softplus function
    to ensure that they remain positive. Samples from the variational
    distribution are generated using the reparameterization trick.

    Args:
        x: Input feature matrix with shape
            ``(num_observations, num_features)``. To include an intercept,
            append a column of ones to this matrix.
        y: Binary target labels encoded as ``-1`` and ``1``, with shape
            ``(num_observations,)``.

    Returns:
        A tuple containing:

        - The variational posterior mean with shape ``(num_features,)``.
        - The variational posterior covariance matrix with shape
          ``(num_features, num_features)``.
    """
    num_params = x.shape[1]
    mu = torch.zeros(num_params, requires_grad=True)
    raw_c = torch.zeros(num_params, num_params, requires_grad=True)

    optimizer = SGD([mu, raw_c], lr=0.0001)

    for epoch in range(1, 1001):
        optimizer.zero_grad()

        lower = torch.tril(raw_c, diagonal=-1)
        diagonal = f.softplus(
            torch.diagonal(raw_c)
        ) + 1e-6

        matrix_c = lower + torch.diag(diagonal)
        cov = matrix_c @ matrix_c.T

        num_mc_samples = 10

        epsilon = torch.randn(num_mc_samples, num_params)
        theta = mu[None, :] + epsilon @ matrix_c.T

        logits = x @ theta.T

        loss = vi_loss(logits, y, mu, cov, x.shape[0])

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss {float(loss.item()):.4f}")

    cov = raw_c @ raw_c.T
    return mu, cov


def bayes_log_reg_loss(
    y_true: torch.Tensor,
    x_input: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
    prior_variance: float = 1.0
):
    """Compute the negative log-posterior for Bayesian logistic regression.

    The function assumes labels encoded as -1 and +1 and places an isotropic
    zero-mean Gaussian prior on the weight vector.

    Args:
        y_true: Target labels with shape ``(num_samples,)``, encoded as
            either -1 or +1.
        x_input: Input features with shape
            ``(num_samples, num_features)``.
        weights: Weight vector with shape ``(num_features,)``.
        bias: Scalar bias parameter.
        prior_variance: Variance of the Gaussian prior over the weights.

    Returns:
        The negative log-posterior up to an additive constant.
    """
    logits = x_input @ weights + bias
    negative_log_likelihood = torch.sum(
        torch.log1p(torch.exp(-y_true * logits))
    )

    negative_log_prior = (
        weights.square().sum() / (2.0 * prior_variance)
    )

    return negative_log_likelihood + negative_log_prior


def bayesian_log_reg_laplace(
    x: torch.Tensor,
    y: torch.Tensor,
    prior_var: float = 1.0
):
    """Fit Bayesian logistic regression using a Laplace approximation.

    The function first finds the maximum a posteriori estimates of the
    logistic regression weights and bias using stochastic gradient descent.
    It then approximates the posterior distribution around the optimum with a
    multivariate Gaussian distribution. The posterior covariance matrix is
    obtained by inverting the Hessian of the negative log-posterior with
    respect to the weight vector.

    Args:
        x: Input feature matrix with shape
            ``(num_observations, num_features)``.
        y: Binary target labels with shape ``(num_observations,)``. The label
            encoding must match the encoding expected by
            ``bayes_log_reg_loss``.
        prior_var: Variance of the zero-mean isotropic Gaussian prior over the
            weight vector. Must be greater than zero.

    Returns:
        A tuple containing:

        - The maximum a posteriori weight vector with shape
          ``(num_features,)``.
        - The maximum a posteriori bias with shape ``(1,)``.
        - The approximate posterior covariance matrix of the weights, with
          shape ``(num_features, num_features)``.
    """
    num_features = x.shape[1]
    w = torch.zeros(num_features, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    optimizer = SGD([w, b], lr=0.001)

    for epoch in range(1, 501):
        optimizer.zero_grad()

        loss = bayes_log_reg_loss(y, x, w, b, prior_var)
        loss.backward()

        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss {float(loss.item()):.4f}")

    probs = torch.sigmoid(x @ w + b)
    fisher = (
        prior_var * torch.eye(num_features) +
        x.T @ ((probs * (1.0 - probs))[:, None] * x)
    )

    posterior_covariance = torch.linalg.inv(fisher)
    return w, b, posterior_covariance
