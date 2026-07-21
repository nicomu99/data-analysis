"""Bayesian Neural Network with Variational Inference."""

import torch
import torch.nn as nn
import torch.nn.functional as f


class BayesianLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight_mean = nn.Parameter(
            torch.randn(out_features, in_features) * 0.1
        )
        self.weight_rho = nn.Parameter(
            torch.full((out_features, in_features), -3.0)
        )

        self.bias_mean = nn.Parameter(
            torch.zeros(out_features)
        )
        self.bias_rho = nn.Parameter(
            torch.full((out_features,), -3.0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute a linear prediction

        For the prediction, a weight vector is first sampled using the
        reparametrization trick.

        Args:
            x: Input tensor with shape ``(batch_size, in_features)``.

        Returns:
            A tuple containing the predicted target value. The tensor has
            shape ``(batch_size, out_features)``.
        """
        weight_std = f.softplus(self.weight_rho) + 1e-8
        bias_std = f.softplus(self.bias_rho) + 1e-8

        weight = self.weight_mean + weight_std * torch.randn_like(self.weight_mean)
        bias = self.bias_mean + bias_std * torch.randn_like(self.bias_mean)

        return x @ weight.T + bias

    def kl_divergence(self) -> float:
        """Computes the KL-Divergence of the weight matrices.

        Returns:
            The KL-divergence between the weight samples and a standard
            Gaussian distribution.
        """
        weight_std = f.softplus(self.weight_rho)
        bias_std = f.softplus(self.bias_rho)

        weight_kl = 0.5 * (
                weight_std.square()
                + self.weight_mean.square()
                - 1.0
                - 2.0 * torch.log(weight_std)
        ).sum()

        bias_kl = 0.5 * (
                bias_std.square()
                + self.bias_mean.square()
                - 1.0
                - 2.0 * torch.log(bias_std)
        ).sum()

        return weight_kl.item() + bias_kl.item()


class BNNVI(nn.Module):
    def __init__(self, in_features: int, out_features: int = 1):
        """Initialize the neural network.

        Args:
            in_features: Number of features in each input observation.
        """
        super().__init__()
        self.shared = BayesianLinear(in_features, 32)

        self.mean_hidden = BayesianLinear(32, 16)
        self.mean_out = BayesianLinear(16, out_features)

        self.var_hidden = BayesianLinear(32, 16)
        self.log_var_out = BayesianLinear(16, out_features)

    def forward(
        self,
        x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the target mean and log-variance.

        Args:
            x: Input tensor with shape ``(batch_size, in_features)``.

        Returns:
            A tuple containing the predicted target means and target
            log-variances. Both tensors have shape ``(batch_size, out_features)``.
        """
        x = f.tanh(self.shared(x))

        mean_features = f.tanh(self.mean_hidden(x))
        var_features = f.tanh(self.var_hidden(x))

        mean_y = self.mean_out(mean_features)
        log_var_y = self.log_var_out(var_features)

        return mean_y, log_var_y

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        num_samples: int = 10
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict the target mean and predictive variances.

        Args:
            x: Input tensor with shape ``(batch_size, in_features)``.
            num_samples: Number of Monte Carlo parameter samples used to estimate
                the predictive distribution.

        Returns:
            A tuple containing:

            - Predictive mean with shape ``(batch_size, out_features)``.
            - Aleatoric variance with shape ``(batch_size, out_features)``.
            - Total predictive variance with shape
              ``(batch_size, out_features)``.

        Raises:
            ValueError: If ``num_samples`` is less than one.
        """
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1.")

        mean_samples, log_var_samples = [], []
        for _ in range(num_samples):
            mean_y, log_var_y = self(x)
            mean_samples.append(mean_y)
            log_var_samples.append(log_var_y)

        mean_samples = torch.stack(mean_samples, dim=0)
        log_var_samples = torch.stack(log_var_samples, dim=0)

        predictive_mean = mean_samples.mean(dim=0)

        aleatoric_var = log_var_samples.exp().mean(dim=0)
        epistemic_var = mean_samples.var(dim=0, correction=0,)
        total_var = aleatoric_var + epistemic_var

        return predictive_mean, aleatoric_var, total_var

    def kl_divergence(self) -> float:
        """Computes the sum of KL-divergences of all layers.

        Returns:
            The sum of KL-divergences.
        """
        return sum(
            module.kl_divergence()
            for module in self.modules()
            if isinstance(module, BayesianLinear)
        )
