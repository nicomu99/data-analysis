"""Bayesian Neural Network based on dropout as variational inference."""

import torch
import torch.nn as nn
import torch.nn.functional as f


class BNNDropout(nn.Module):
    """Neural network that uses dropout during inference.

    The network uses shared hidden layers followed by two separate output
    heads. One head predicts the conditional mean, while the other predicts
    the logarithm of the conditional variance. The model performs several
    forward passes and keeps dropout enabled during inference, which can
    be seen as a type of variational inference.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        dropout_p: float = 0.3
    ):
        """Initialize the neural network.

        Args:
            in_features: Number of features in each input observation.
            out_features: Number of output features.
            dropout_p: Dropout probability.
        """
        super().__init__()
        self.shared = nn.Linear(in_features, 32)
        self.dropout = nn.Dropout(p=dropout_p)

        self.mean_hidden = nn.Linear(32, 16)
        self.mean_out = nn.Linear(16, out_features)

        self.var_hidden = nn.Linear(32, 16)
        self.log_var_out = nn.Linear(16, out_features)

    def train(self, mode=True):
        super().train(mode)

        def _enable_dropout(module: nn.Module):
            if isinstance(module, nn.Dropout):
                module.train()

        if not mode:
            self.apply(_enable_dropout)
        return self

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
        x = f.softplus(self.shared(x))
        x = self.dropout(x)

        mean_features = f.softplus(self.mean_hidden(x))
        var_features = f.softplus(self.var_hidden(x))

        mean_features = self.dropout(mean_features)
        var_features = self.dropout(var_features)

        mean_y = self.mean_out(mean_features)
        log_var_y = self.log_var_out(var_features)

        return mean_y, log_var_y

    def predict(
        self,
        x: torch.Tensor,
        num_passes: int = 10
    ):
        """Predict target means and predictive variances.

        The method combines the observation variance predicted by the variance
        head with the epistemic variance obtained from the forward passes.

        Args:
            x: Input tensor with shape ``(num_samples, in_features)``.
            num_passes: Number of forward passes.

        Returns:
            A tuple containing:

            - Predicted target means with shape ``(num_samples, 1)``.
            - Predicted aleatoric variances with shape ``(num_samples, 1)``.
            - Total predictive variances with shape ``(num_samples, 1)``.
        """
        mean_samples, log_var_samples = [], []
        for i in range(num_passes):
            mean_y, log_var_y = self(x)
            mean_samples.append(mean_y)
            log_var_samples.append(log_var_y)

        mean_samples = torch.stack(mean_samples, dim=0)
        log_var_samples = torch.stack(log_var_samples, dim=0)

        predictive_mean = mean_samples.mean(dim=0)

        aleatoric_var = log_var_samples.exp().mean(dim=0)
        epistemic_var = mean_samples.var(dim=0, unbiased=True)
        total_var = aleatoric_var + epistemic_var

        return predictive_mean, aleatoric_var, total_var
