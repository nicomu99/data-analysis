"""BNN with MCMC sampling."""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as f


class BNNMCMC(nn.Module):
    """Bayesian neural network trained using MCMC weight sampling.

    The network predicts both the conditional mean and log-variance of a
    Gaussian observation model. Posterior uncertainty is approximated by
    storing multiple network parameter states produced by an MCMC method such
    as Stochastic Gradient Langevin Dynamics.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        prior_variance: Variance of the independent zero-mean Gaussian prior
            placed on every trainable parameter.

    Attributes:
        prior_variance_: Variance of the Gaussian parameter prior.
        weight_samples_: Stored posterior samples of the model parameters.
            Each dictionary contains one complete model state.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        prior_variance: float = 1.0
    ):
        """Initialize the Bayesian neural network.

        Args:
            in_features: Number of input features.
            out_features: Number of predicted target features.
            prior_variance: Variance of the independent zero-mean Gaussian
                prior placed on each trainable parameter.
        """
        super().__init__()
        self.shared = nn.Linear(in_features, 32)

        self.mean_features = nn.Linear(32, 16)
        self.mean_head = nn.Linear(16, out_features)

        self.log_var_features = nn.Linear(32, 16)
        self.log_var_head = nn.Linear(16, out_features)

        self.prior_variance_ = prior_variance
        self.weight_samples_: list[dict[str, torch.Tensor]] = []

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the conditional mean and log-variance.

        Args:
            x: Input tensor with shape ``(batch_size, in_features)``.

        Returns:
            A tuple containing:

            - The predicted conditional mean with shape
              ``(batch_size, out_features)``.
            - The predicted conditional log-variance with shape
              ``(batch_size, out_features)``.
        """
        x = f.sigmoid(self.shared(x))

        mean_features = f.sigmoid(self.mean_features(x))
        mean = self.mean_head(mean_features)

        log_var_features = f.sigmoid(self.log_var_features(x))
        log_var = self.log_var_head(log_var_features)

        return mean, log_var

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute posterior predictive statistics from stored weight samples.

        Each stored parameter state is loaded into the network and used to
        predict a conditional mean and variance. The predictive uncertainty is
        decomposed into aleatoric and epistemic components.

        The original parameter state of the model is restored after prediction.

        Args:
            x: Input tensor with shape ``(batch_size, in_features)``.

        Returns:
            A tuple containing:

            - The posterior predictive mean.
            - The aleatoric variance, calculated as the mean predicted
              observation variance across weight samples.
            - The total predictive variance, calculated as the sum of
              aleatoric and epistemic variance.

            All returned tensors have shape
            ``(batch_size, out_features)``.

        Raises:
            RuntimeError: If no weight samples have been stored.
        """
        if not self.weight_samples_:
            raise RuntimeError("No weight samples have been stored.")

        original_state = {
            name: tensor.detach().clone()
            for name, tensor in self.state_dict().items()
        }

        mean_samples, log_var_samples = [], []
        for i in range(len(self.weight_samples_)):
            self.load_weight_sample(i)
            mean_y, log_var_y = self(x)
            mean_samples.append(mean_y)
            log_var_samples.append(log_var_y)

        self.load_state_dict(original_state)

        mean_samples = torch.stack(mean_samples, dim=0)
        log_var_samples = torch.stack(log_var_samples, dim=0)

        predictive_mean = mean_samples.mean(dim=0)

        aleatoric_var = log_var_samples.exp().mean(dim=0)
        epistemic_var = mean_samples.var(dim=0, unbiased=True)
        total_var = aleatoric_var + epistemic_var

        return predictive_mean, aleatoric_var, total_var

    def negative_log_prior(self) -> torch.Tensor | Literal[0]:
        """Calculate the negative log-prior of the model parameters.

        All trainable parameters are assumed to have independent zero-mean
        Gaussian priors with variance ``prior_variance_``. Terms that are
        constant with respect to the parameters are omitted.

        Returns:
            The negative log-prior summed over all trainable parameters. The
            integer ``0`` is returned if the model has no trainable parameters.
        """
        return sum(
            parameter.square().sum() / (2.0 * self.prior_variance_)
            for parameter in self.parameters()
        )

    def store_weight_sample(self) -> None:
        """Store a copy of the current model parameter state.

        The tensors are detached from the computation graph, moved to the CPU,
        and cloned before being added to ``weight_samples_``. This prevents
        later parameter updates from modifying the stored sample.
        """
        sample = {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.state_dict().items()
        }
        self.weight_samples_.append(sample)

    def load_weight_sample(self, index: int) -> None:
        """Load a stored parameter sample into the model.

        Args:
            index: Index of the parameter sample in ``weight_samples_``.

        Raises:
            IndexError: If ``index`` does not refer to an existing sample.
        """
        self.load_state_dict(self.weight_samples_[index])

    def clear_weight_samples(self) -> None:
        """Remove all stored parameter samples."""
        self.weight_samples_.clear()
