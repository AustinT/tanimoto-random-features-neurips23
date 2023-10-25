"""Code for GPs."""
from __future__ import annotations

import logging
from typing import Optional, Union

import botorch
import gpytorch
import numpy as np
import torch
from gpytorch.kernels import Kernel, ScaleKernel
from gpytorch.models import ApproximateGP, ExactGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

from .tanimoto_functions import batch_tdp_sim, batch_tmm_sim

logger = logging.getLogger(__name__)


class TMM_Kernel(Kernel):
    is_stationary = False
    has_lengthscale = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            assert x1.size() == x2.size() and torch.equal(x1, x2)
            return torch.ones(*x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device)
        else:
            return batch_tmm_sim(x1, x2)


class TDP_Kernel(Kernel):
    is_stationary = False
    has_lengthscale = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            assert x1.size() == x2.size() and torch.equal(x1, x2)
            return torch.ones(*x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device)
        return batch_tdp_sim(x1, x2)


class TanimotoKernelGP(
    ExactGP,
    botorch.models.gpytorch.GPyTorchModel,
):
    _num_outputs = 1  # looks like botorch needs this

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        kernel: Union[str, Kernel],
        mean_obj: gpytorch.means.Mean,
        likelihood: Optional[gpytorch.likelihoods.Likelihood] = None,
    ):
        # Create likelihood
        likelihood = likelihood or gpytorch.likelihoods.GaussianLikelihood()

        botorch.models.gpytorch.GPyTorchModel.__init__(self)
        ExactGP.__init__(self, train_x, train_y, likelihood)

        # Set mean and covar attributes
        self.mean_module = mean_obj
        if isinstance(kernel, Kernel):
            self.covar_module = kernel
        elif kernel == "T_DP":
            self.covar_module = ScaleKernel(TDP_Kernel())
        elif kernel == "T_MM":
            self.covar_module = ScaleKernel(TMM_Kernel())
        else:
            raise NotImplementedError(kernel)

    def forward(self, x):
        # Normal mean + covar
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class CustomKernelSVGP(ApproximateGP):
    def __init__(
        self,
        kernel_obj: Kernel,
        inducing_points: torch.Tensor,
        mean_obj: Optional[gpytorch.means.Mean],
        variational_distribution_class=CholeskyVariationalDistribution,
    ):
        # Init approx GP
        variational_distribution = variational_distribution_class(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)

        # Mean and covar modules
        self.mean_module = mean_obj
        self.covar_module = kernel_obj

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def inducing_point_parameters(self) -> list[torch.nn.parameter.Parameter]:
        return [self.variational_strategy.inducing_points]

    def set_inducing_points_requires_grad(self, value: bool) -> None:
        for param in self.inducing_point_parameters():
            param.requires_grad_(value)


def get_gp_hyperparameters(gp_model) -> dict[str, float]:
    out = dict(
        outputscale=gp_model.covar_module.outputscale.item(),
        noise=gp_model.likelihood.noise.item(),
    )
    if hasattr(gp_model.mean_module, "constant"):
        out["mean"] = gp_model.mean_module.constant.item()
    return out


def fit_exact_gp_hyperparameters_scipy(gp_model: ExactGP, **kwargs):
    """Optimize train MLL to fit GP hyperparameters"""

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
    gp_model.train()
    mll.train()

    opt_res = botorch.optim.fit.fit_gpytorch_scipy(mll, **kwargs)
    logger.debug(opt_res[1])  # Log fit results
    return opt_res


def batch_predict_mu_std_numpy(
    gp_model: gpytorch.models.GP,
    x: Union[torch.Tensor, np.ndarray],
    batch_size: int = 2**15,  # about 32 k
    include_std: bool = True,
    device=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the [marginal] mean and standard deviation of the GP posterior at the given points."""

    # Clear all caches and put into eval mode
    gp_model.train()
    gp_model.eval()

    # Do batch prediction
    mu = []
    std = []
    with gpytorch.settings.fast_computations(False, False, False), torch.no_grad():
        for batch_start in range(0, len(x), batch_size):
            batch_end = batch_start + batch_size
            x_batch = torch.as_tensor(x[batch_start:batch_end], device=device)
            output = gp_model(x_batch)
            mu_batch = output.mean.detach().cpu().numpy()
            if include_std:
                std_batch = output.stddev.detach().cpu().numpy()
            else:
                std_batch = np.zeros_like(mu_batch)
            mu.append(mu_batch)
            std.append(std_batch)

    # Concatenate batches and return
    mu = np.concatenate(mu, axis=0)
    std = np.concatenate(std, axis=0)
    return mu, std
