"""Gaussian Process implementation."""

import numpy as np


class RBF:
    """Radial basis function kernel.

    Args:
        h: Length-scale parameter controlling how quickly similarity decreases
            with distance. Must be greater than zero.
    """
    def __init__(self, h: float = 1.0):
        """Initialize the radial basis function kernel.

        Args:
            h: Length-scale parameter controlling the kernel width. Must be
                greater than zero.
        """
        self.h_ = h ** 2

    def __call__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *args,
        **kwargs
    ) -> np.ndarray:
        """Compute the RBF kernel matrix.

        Args:
            x: First input array with shape ``(n_samples_x, n_features)``.
            y: Second input array with shape ``(n_samples_y, n_features)``.

        Returns:
            Kernel matrix with shape ``(n_samples_x, n_samples_y)``, where
            element ``(i, j)`` contains the similarity between ``x[i]`` and
            ``y[j]``.
        """
        dist = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=2) ** 2
        return np.exp(-dist / self.h_)


class Linear:
    """Linear kernel."""
    def __call__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *args,
        **kwargs
    ) -> np.ndarray:
        """Compute the linear kernel matrix.

        Args:
            x: First input array with shape ``(n_samples_x, n_features)``.
            y: Second input array with shape ``(n_samples_y, n_features)``.

        Returns:
            Kernel matrix with shape ``(n_samples_x, n_samples_y)``, where
            element ``(i, j)`` contains the dot product between ``x[i]`` and
            ``y[j]``.
        """
        return x @ y.T


class Exponential:
    """Exponential kernel.

    Args:
        h: Length-scale parameter controlling how quickly similarity decreases
            with distance. Must be greater than zero.
    """
    def __init__(self, h: float = 1.0):
        """Initialize the exponential kernel.

        Args:
            h: Length-scale parameter controlling the kernel width. Must be
                greater than zero.
        """
        self.h_ = h

    def __call__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *args,
        **kwargs
    ) -> np.ndarray:
        """Compute the exponential kernel matrix.

        Args:
            x: First input array with shape ``(n_samples_x, n_features)``.
            y: Second input array with shape ``(n_samples_y, n_features)``.

        Returns:
            Kernel matrix with shape ``(n_samples_x, n_samples_y)``, where
            element ``(i, j)`` contains the similarity between ``x[i]`` and
            ``y[j]``.
        """
        dist = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=2)
        return np.exp(-dist / self.h_)


class GP:
    """Gaussian process regression model.

    The model uses a configurable covariance kernel to define similarities
    between input samples. Observation noise is represented by the noise
    standard deviation ``sigma_n``.

    Supported kernels are:

    - ``"rbf"``: Radial basis function kernel.
    - ``"linear"``: Linear dot-product kernel.
    - ``"exponential"``: Exponential distance kernel.

    Args:
        kernel: Name of the covariance kernel to use.
        sigma_n: Standard deviation of the observation noise. Must be
            non-negative.
        h: Length-scale parameter used by the RBF and exponential kernels.
            Must be greater than zero.

    Attributes:
        kernel_: Kernel object used to compute covariance matrices.
        sigma_n_: Standard deviation of the observation noise.
        kernel_coef_: Coefficients used to compute the predictive mean after
            fitting. Set to ``None`` before fitting.
        mu_coef_: Coefficients associated with the model's mean function.
            Set to ``None`` before fitting.
        x_train_: Training input array. Set to ``None`` before fitting.

    Raises:
        ValueError: If ``kernel`` is not one of the supported kernel names.
    """
    def __init__(
        self,
        kernel: str = "rbf",
        sigma_n: float = 0.1,
        h: float = 1
    ):
        """Initialize the Gaussian process model.

        Args:
            kernel: Name of the covariance kernel to use. Supported values are
                ``"rbf"``, ``"linear"``, and ``"exponential"``.
            sigma_n: Standard deviation of the observation noise. Must be
                non-negative.
            h: Length-scale parameter used by the RBF and exponential kernels.
                Must be greater than zero.

        Raises:
            ValueError: If the specified kernel is unknown.
        """
        _kernel_map = {
            "rbf": RBF(h),
            "linear": Linear(),
            "exponential": Exponential(h)
        }
        if kernel not in _kernel_map:
            raise ValueError("kernel unknown.")
        self.kernel_ = _kernel_map[kernel]
        self.sigma_n_ = float(sigma_n)

        self.kernel_coef_ = None
        self.mu_coef_ = None
        self.x_train_ = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GP":
        """Fit the Gaussian Processes.

        The model assumes a zero-prior.

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
        if x.ndim != 2:
            raise ValueError("x must be a two-dimensional array.")

        if x.shape[0] == 0:
            raise ValueError("x and y must contain at least one sample.")

        if x.shape[0] != y.shape[0]:
            raise ValueError(
                "x and y must contain the same number of samples."
            )
        num_samples = x.shape[0]

        self.x_train_ = x

        self.kernel_coef_ = np.linalg.inv(
            self.kernel_(self.x_train_, self.x_train_) + self.sigma_n_ ** 2 * np.eye(num_samples)
        )
        self.mu_coef_ = self.kernel_coef_ @ y

        return self

    def predict(self, x: np.ndarray):
        """Predict target means and their joint predictive covariance.

        Args:
            x: Feature matrix with shape ``(num_samples, num_features)``.

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
        if self.mu_coef_ is None or self.kernel_coef_ is None:
            raise RuntimeError("The model must be fitted before prediction.")

        if x.ndim != 2:
            raise ValueError("x must be a two-dimensional array.")

        expected_features = self.x_train_.shape[1]
        if x.shape[1] != expected_features:
            raise ValueError(
                f"x must contain {expected_features} features, "
                f"but received {x.shape[1]}."
            )
        y_kernel = self.kernel_(x, self.x_train_)
        mu = y_kernel @ self.mu_coef_
        cov = (
                self.kernel_(x, x)
                - y_kernel @ self.kernel_coef_ @ y_kernel.T
        )
        cov += np.eye(len(x)) * self.sigma_n_ ** 2

        return mu, cov
