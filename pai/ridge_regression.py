"""Ridge regression implemented with mini-batch gradient descent."""

import numpy as np


class RidgeRegression:
    """Linear ridge regression model trained with mini-batch gradient descent.

    Ordinary least squares regression with L2 regularization to reduce
    overfitting to improve numerical stability when input features are
    correlated.

    Attributes:
        lam: Strength of the L2 regularization penalty.
        max_iter: Number of gradient descent iterations.
        lr: Learning rate used for gradient descent updates.
        w: Learned model parameters. The first value is the intercept, and the
            remaining values are feature coefficients. This is ``None`` before the
            model is fitted.
    """
    def __init__(
        self,
        lam: float = 1.0,
        max_iter: int = 10_000,
        lr: float = 0.01
    ):
        self.lam = float(lam)
        self.max_iter = int(max_iter)
        self.lr = float(lr)
        self.w = None

    def gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Computes the gradient of the ridge regression objective.

        The first weight is assumed to be the intercept and is not included in
        the regularization term.

        Args:
            x: Design matrix with shape ``(n_samples, n_features + 1)``. The
                first column must contain ones for the intercept.
            y: Target values with shape ``(n_samples,)``.

        Returns:
            Gradient vector with the same shape as ``self.w``.

        Raises:
            RuntimeError: If the model parameters have not been initialized.
        """
        if self.w is None:
            raise RuntimeError("Model has not been initialized.")

        n = x.shape[0]
        error = x @ self.w - y
        gradient = (2 / n) * x.T @ error

        regularization = 2 * self.lam * self.w
        regularization[0] = 0

        return gradient + regularization

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeRegression":
        """Fits the ridge regression model to the training data.

        The model parameters are optimized using mini-batch gradient descent.
        Each iteration randomly samples up to 10 training observations. An
        intercept column is automatically added to the input matrix.

        Args:
            x: Training feature matrix with shape ``(n_samples, n_features)``.
            y: Training target values with shape ``(n_samples,)`` or
                ``(n_samples, 1)``.

        Returns:
            The fitted model instance.

        Raises:
            ValueError: If ``x`` is not a two-dimensional array, if ``x`` and
            ``y`` contain different numbers of samples, or if the training
            dataset is empty.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        num_samples, num_features = x.shape
        batch_size = min(10, num_samples)

        self.w = np.random.random(num_features + 1)
        for _ in range(self.max_iter):
            # pick a random subset of the data
            sample = np.random.choice(np.arange(num_samples), batch_size)
            batch_x = np.hstack((np.ones((batch_size, 1)), x[sample]))
            batch_y = y[sample]

            # compute loss
            g = self.gradient(batch_x, batch_y)

            # update
            self.w -= self.lr * g
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicts target values for the given input samples.

        An intercept column is automatically added to the input matrix before
        applying the learned model parameters.

        Args:
            x: Feature matrix with shape ``(n_samples, n_features)``.

        Returns:
            Predicted target values with shape ``(n_samples,)``.

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If ``x`` is not two-dimensional or contains a different
                number of features than the training data. """
        if self.w is None:
            raise RuntimeError("Model has not been fitted.")

        x = np.asarray(x, dtype=float)
        x = np.hstack((np.ones((x.shape[0], 1)), x))

        return x @ self.w
