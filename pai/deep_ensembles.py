import torch
import torch.nn as nn
import torch.nn.functional as f


class DeepEnsembleBaseNN(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
    ):
        """Initialize the neural network.

        Args:
            in_features: Number of features in each input observation.
            out_features: Number of output features.
        """
        super().__init__()
        self.shared = nn.Linear(in_features, 32)

        self.mean_hidden = nn.Linear(32, 16)
        self.mean_out = nn.Linear(16, out_features)

        self.var_hidden = nn.Linear(32, 16)
        self.log_var_out = nn.Linear(16, out_features)

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

        mean_features = f.softplus(self.mean_hidden(x))
        var_features = f.softplus(self.var_hidden(x))

        mean_y = self.mean_out(mean_features)
        log_var_y = self.log_var_out(var_features)

        return mean_y, log_var_y


class DeepEnsemble(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        num_models: int = 10,
    ):
        """Initialize the neural network.

        Args:
            in_features: Number of features in each input observation.
            out_features: Number of output features.
            num_models: Number of deep ensemble models.
        """
        super().__init__()
        self.models = nn.ModuleList([
            DeepEnsembleBaseNN(in_features, out_features)
            for _ in range(num_models)
        ])

    def forward(
        self,
        x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the target mean and log-variance.

        This function should not be used!

        Args:
            x: Input tensor with shape ``(batch_size, in_features)``.

        Returns:
            A tuple containing the predicted target means and target
            log-variances. Both tensors have shape ``(batch_size, 1)``.
        """
        mean_y, log_var_y = self.models[0](x)
        return mean_y, log_var_y

    def predict(
        self,
        x: torch.Tensor,
    ):
        """Predict target means and predictive variances.

        The method combines the observation variance predicted by the variance
        head with the epistemic variance obtained from the forward passes.

        Args:
            x: Input tensor with shape ``(num_samples, in_features)``.

        Returns:
            A tuple containing:

            - Predicted target means with shape ``(num_samples, 1)``.
            - Predicted aleatoric variances with shape ``(num_samples, 1)``.
            - Total predictive variances with shape ``(num_samples, 1)``.
        """
        mean_samples, log_var_samples = [], []
        for m in self.models:
            mean_y, log_var_y = m(x)
            mean_samples.append(mean_y)
            log_var_samples.append(log_var_y)

        mean_samples = torch.stack(mean_samples, dim=0)
        log_var_samples = torch.stack(log_var_samples, dim=0)

        predictive_mean = mean_samples.mean(dim=0)

        aleatoric_var = log_var_samples.exp().mean(dim=0)
        epistemic_var = mean_samples.var(dim=0, unbiased=True)
        total_var = aleatoric_var + epistemic_var

        return predictive_mean, aleatoric_var, total_var
