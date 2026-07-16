"""Gaussian Process implementation."""

import numpy as np


class RBF:
    """Radial basis function kernel.

    Args:
        h: Length-scale parameter controlling how quickly similarity decreases
            with distance. Must be greater than zero.
    """
    def __init__(self, h: float = 1.0, num_features: int = 1000):
        """Initialize the radial basis function kernel.

        Args:
            h: Length-scale parameter controlling the kernel width. Must be
                greater than zero.
            num_features: If the kernel is approximated, this is the number of
                fourier transform features.
        """
        if h <= 0:
            raise ValueError("h must be greater than zero.")

        self.h_ = float(h)
        self.num_features_ = num_features
        self.approximate_ = True

        self.w_ = None
        self.b_ = None

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

    def fit_features(self, input_dim: int) -> None:
        """Sample random Fourier feature parameters.

        Args:
            input_dim: Number of dimensions in each input sample.

        Returns:
            A tuple containing:
                - Frequencies with shape
                  ``(input_dim, num_features)``.
                - Random phases with shape ``(num_features,)``.
        """
        self.w_ = (
            np.sqrt(2) / self.h_
            * np.random.randn(input_dim, self.num_features_)
        )
        self.b_ = 2 * np.pi * np.random.rand(self.num_features_)

    def transform(self, x: np.ndarray) -> np.ndarray:
        input_dim = x.shape[1]
        if self.w_ is None or self.b_ is None:
            self.fit_features(input_dim)

        return (
            np.sqrt(2 / self.num_features_)
            * np.cos(x @ self.w_ + self.b_)
        )


class Linear:
    """Linear kernel."""
    def __init__(self):
        self.approximate_ = False

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
        self.approximate_ = False

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
        *,
        sigma_n: float = 0.1,
        h: float = 1,
        approximate: bool = False,
        num_features: int = 100
    ):
        """Initialize the Gaussian process model.

        Args:
            kernel: Name of the covariance kernel to use. Supported values are
                ``"rbf"``, ``"linear"``, and ``"exponential"``.
            sigma_n: Standard deviation of the observation noise. Must be
                non-negative.
            h: Length-scale parameter used by the RBF and exponential kernels.
                Must be greater than zero.
            approximate: Whether to approximate the kernel using random fourier
                features. Can only be applied for shift-invariant kernels.
                Defaults to False.
            num_features: If the kernel is to be approximated, this is the
                number of fourier features. Defaults to 100.

        Raises:
            ValueError: If the specified kernel is unknown.
        """
        _kernel_map = {
            "rbf": RBF(h, num_features),
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
        self.z_train_ = None
        self.approximate_ = approximate
        if not self.kernel_.approximate_ and self.approximate_:
            self.approximate_ = False

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

        if self.approximate_ and self.kernel_.approximate_:
            self.z_train_ = self.kernel_.transform(x)
            self.kernel_coef_ = np.linalg.inv(
                np.dot(self.z_train_.T, self.z_train_)
                + self.sigma_n_ ** 2 * np.eye(self.z_train_.shape[1])
            )
            self.mu_coef_ = self.kernel_coef_ @ self.z_train_.T @ y
        else:
            kernel_matrix = self.kernel_(self.x_train_, self.x_train_)
            self.kernel_coef_ = np.linalg.inv(
                kernel_matrix + self.sigma_n_ ** 2 * np.eye(num_samples)
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

        if self.approximate_:
            z_kernel = self.kernel_.transform(x)
            mu = z_kernel @ self.mu_coef_
            cov = self.sigma_n_ ** 2 * (
                 z_kernel @ self.kernel_coef_ @ z_kernel.T
            )
            cov = (cov + cov.T) / 2
            cov += np.eye(len(x)) * self.sigma_n_ ** 2
        else:
            y_kernel = self.kernel_(x, self.x_train_)
            mu = y_kernel @ self.mu_coef_
            cov = (
                    self.kernel_(x, x)
                    - y_kernel @ self.kernel_coef_ @ y_kernel.T
            )
            cov = (cov + cov.T) / 2
            cov += np.eye(len(x)) * self.sigma_n_ ** 2

        return mu, cov
