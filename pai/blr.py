import numpy as np


class BLR:
    def __init__(
        self,
        alpha: float = 1.0,
        sigma_n: float = 1.0
    ):
        self.alpha = float(alpha)
        self.sigma_n = float(sigma_n)
        self.sigma_p = self.sigma_n / self.alpha
        self.alpha = self.sigma_n / self.sigma_n
        self.mu = None
        self.cov = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "BLR":
        num_samples, num_features = x.shape[0], x.shape[1] + 1

        x = np.hstack((np.ones((num_samples, 1)), np.asarray(x, dtype=float)))
        y = np.asarray(y, dtype=float).ravel()

        temp = np.linalg.inv(x.T @ x + self.alpha * np.eye(num_features))
        self.mu = temp @ x.T @ y
        self.cov = self.sigma_n * temp

        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.hstack((np.ones((x.shape[0], 1)), np.asarray(x, dtype=float)))
        mu = x @ self.mu
        cov = x @ self.cov @ x.T + self.sigma_n

        return mu, cov
