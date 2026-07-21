"""Neural network that predicts mean and log variance."""

import torch
import torch.nn as nn
import torch.nn.functional as f


class BNNLA(nn.Module):
    """Neural network that predicts a Gaussian mean and log-variance.

    The network uses shared hidden layers followed by two separate output
    heads. One head predicts the conditional mean, while the other predicts
    the logarithm of the conditional variance.

    Despite the class name, this model does not use Bayesian weights. It models
    input-dependent aleatoric uncertainty through its variance prediction.

    Args:
        in_features: Number of input features.
        lam: Inverse of the prior variance.
    """
    def __init__(self, in_features: int, lam: float):
        """Initialize the neural network.

        Args:
            in_features: Number of features in each input observation.
            lam: Inverse of the prior variance.
        """
        super().__init__()
        self.shared = nn.Linear(in_features, 32)

        self.mean_hidden = nn.Linear(32, 16)
        self.mean_out = nn.Linear(16, 1)

        self.var_hidden = nn.Linear(32, 16)
        self.log_var_out = nn.Linear(16, 1)

        self.sigma_ = None
        self.lam_ = lam

    def forward(
        self,
        x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the target mean and log-variance.

        Args:
            x: Input tensor with shape ``(batch_size, in_features)``.

        Returns:
            A tuple containing the predicted target means and target
            log-variances. Both tensors have shape ``(batch_size, 1)``.
        """
        x = f.relu(self.shared(x))

        mean_features = f.tanh(self.mean_hidden(x))
        var_features = f.tanh(self.var_hidden(x))

        mean_y = self.mean_out(mean_features)
        log_var_y = self.log_var_out(var_features)

        return mean_y, log_var_y

    def predict(self, x: torch.Tensor):
        """Predict target means and predictive variances.

        The method combines the observation variance predicted by the variance
        head with the epistemic variance obtained from the last-layer Laplace
        approximation.

        Args:
            x: Input tensor with shape ``(num_samples, in_features)``.

        Returns:
            A tuple containing:

            - Predicted target means with shape ``(num_samples, 1)``.
            - Predicted aleatoric variances with shape ``(num_samples, 1)``.
            - Total predictive variances with shape ``(num_samples, 1)``.

        Raises:
            RuntimeError: If the last-layer covariance matrix has not yet been
                constructed.
        """
        if self.sigma_ is None:
            raise RuntimeError("Covariance matrix not constructed yet")

        x = f.relu(self.shared(x))

        mean_features = f.tanh(self.mean_hidden(x))
        var_features = f.tanh(self.var_hidden(x))

        mean_y = self.mean_out(mean_features)
        log_var_y = self.log_var_out(var_features)

        phi_star = torch.cat((mean_features, torch.ones(x.shape[0], 1)), 1)

        aleatoric_var = torch.exp(log_var_y)

        epistemic_var = torch.sum(
            (phi_star @ self.sigma_) * phi_star,
            dim=1,
            keepdim=True,
        )
        total_var = aleatoric_var + epistemic_var
        return mean_y, aleatoric_var, total_var

    def covariance(self, x: torch.Tensor):
        """Construct the last-layer Laplace covariance matrix.

        The covariance is calculated for the parameters of the mean output layer
        while treating all preceding network layers and the variance head as
        fixed at their MAP estimates. The predicted observation precisions are
        used to weight the hidden feature vectors.

        The resulting covariance matrix is stored in ``self.sigma_``.

        Args:
            x: Training input tensor with shape
                ``(num_samples, in_features)``.
        """
        x = f.relu(self.shared(x))

        mean_features = f.tanh(self.mean_hidden(x))
        var_features = f.tanh(self.var_hidden(x))

        log_var_y = self.log_var_out(var_features)
        precision_w = torch.diag(torch.exp(-log_var_y).reshape((log_var_y.shape[0])))

        phi = torch.cat((mean_features, torch.ones(x.shape[0], 1)), 1)

        self.sigma_ = torch.linalg.inv(
            phi.T @ precision_w @ phi + self.lam_ * torch.eye(phi.shape[1])
        )
