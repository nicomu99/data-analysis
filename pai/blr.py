import numpy as np


class BLR:
    def __init__(self, lam: float = 1.0):
        self.lam = float(lam)
        self.sigma_n2_ = None
        self.sigma_p2_ = None
        self.mu_ = None
        self.cov_ = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "BLR":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        num_samples, num_features = x.shape[0], x.shape[1] + 1  # added for intercept
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
        x = np.hstack((np.ones((x.shape[0], 1)), np.asarray(x, dtype=float)))
        mu = x @ self.mu_
        cov = x @ self.cov_ @ x.T + self.sigma_n2_

        return mu, cov
