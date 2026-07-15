"""Bayesian linear regression with a Gaussian prior."""

import numpy as np


class BLR:
    """Bayesian linear regression with a Gaussian weight prior.

    An intercept column is added automatically. The current implementation
    applies the prior, and therefore regularization, to the intercept as well.

    Args:
        lam: Ratio between the observation noise variance and the prior
            variance. Must be positive.

    Attributes:
        lam: Regularization parameter
            :math:`\\lambda = \\sigma_n^2 / \\sigma_p^2`.
        sigma_n2_: Estimated observation noise variance. Set after fitting.
        sigma_p2_: Estimated prior variance. Set after fitting.
        mu_: Posterior mean of the regression weights. Set after fitting.
        cov_: Posterior covariance matrix of the regression weights. Set after
            fitting.
        """

    def __init__(self, lam: float = 1.0):
        """Initialize the Bayesian linear regression model.

        Args:
            lam: Ratio between the observation noise variance and the prior
                variance. Must be positive.

        Raises:
            ValueError: If ``lam`` is not positive.
        """
        if lam <= 0:
            raise ValueError("lam must be greater than zero.")

        self.lam = float(lam)
        self.sigma_n2_ = None
        self.sigma_p2_ = None
        self.mu_ = None
        self.cov_ = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "BLR":
        """Fit the Bayesian linear regression model.

        The method computes the posterior mean and covariance of the regression
        weights. The observation noise variance is estimated using the mean
        squared training residual, and the prior variance is then obtained from
        the relationship

        Args:
            x: Training feature matrix with shape
                ``(num_samples, num_features)``.
            y: Training target values with shape ``(num_samples,)`` or
                ``(num_samples, 1)``.

        Returns:
            The fitted model instance.

        Raises:
            ValueError: If ``x`` is not two-dimensional, if the number of
                samples in ``x`` and ``y`` differs, or if the training data is
                empty.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if x.ndim != 2:
            raise ValueError("x must be a two-dimensional array.")

        if x.shape[0] == 0:
            raise ValueError("x and y must contain at least one sample.")

        if x.shape[0] != y.shape[0]:
            raise ValueError(
                "x and y must contain the same number of samples."
            )

        num_samples = x.shape[0]
        num_features = x.shape[1] + 1  # added for intercept
        x = np.hstack((np.ones((num_samples, 1)), np.asarray(x, dtype=float)))
        y = np.asarray(y, dtype=float).ravel()

        temp = np.linalg.inv(x.T @ x + self.lam * np.eye(num_features))
        self.mu_ = temp @ x.T @ y

        # Estimate variance
        residuals = y - x @ self.mu_
        self.sigma_n2_ = np.mean(residuals ** 2)
        self.sigma_p2_ = self.sigma_n2_ / self.lam

        self.cov_ = self.sigma_n2_ * temp

        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict target means and their joint predictive covariance.

        The predictive covariance includes both epistemic uncertainty from the
        posterior distribution over the weights and aleatoric uncertainty from
        the observation noise.

        Args:
            x: Feature matrix with shape
                ``(num_samples, num_features)``.

        Returns:
            A tuple containing:

            - Predictive means with shape ``(num_samples,)``.
            - Joint predictive covariance matrix with shape
              ``(num_samples, num_samples)``.

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If ``x`` is not two-dimensional or has a different
                number of features than the training data.
        """
        if self.mu_ is None or self.cov_ is None:
            raise RuntimeError("The model must be fitted before prediction.")

        x = np.asarray(x, dtype=float)

        if x.ndim != 2:
            raise ValueError("x must be a two-dimensional array.")

        x = np.hstack((np.ones((x.shape[0], 1)), x))

        expected_features = self.mu_.shape[0]
        if x.shape[1] != expected_features:
            raise ValueError(
                f"x must contain {expected_features} features, "
                f"but received {x.shape[1]}."
            )

        mu = x @ self.mu_
        cov = x @ self.cov_ @ x.T + self.sigma_n2_

        return mu, cov
