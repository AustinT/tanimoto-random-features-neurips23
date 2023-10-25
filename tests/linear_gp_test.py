import gpytorch
import pytest
import torch

from trf23.linear_gp import (
    linear_gp_mll,
    linear_gp_predict,
)
from trf23.tanimoto_gp import TanimotoKernelGP


@pytest.mark.parametrize("k_var", [0.1, 0.5, 2.0])  # test different kernel variances
@pytest.mark.parametrize("n_var", [0.1, 0.5, 2.0])  # test different noise variances
def test_linear_gp_predict(k_var, n_var):
    # Random data
    D = 4
    x_train = torch.randn(4, D)
    y_train = torch.mean(x_train**2, dim=-1)
    x_test = torch.randn(5, D)

    # Exact GP predictions
    gp_exact = TanimotoKernelGP(
        train_x=x_train,
        train_y=y_train,
        kernel=gpytorch.kernels.LinearKernel(),
        mean_obj=gpytorch.means.ZeroMean(),
    )
    gp_exact.covar_module.variance = k_var
    gp_exact.likelihood.noise = n_var
    gp_exact.eval()
    with torch.no_grad(), gpytorch.settings.fast_computations(False, False, False):
        output = gp_exact(x_test)
        y_test = output.mean
        y_test_covar = output.covariance_matrix.to_dense()

    # Linear GP predictions
    y_test2, y_test_covar2 = linear_gp_predict(
        x_train=x_train,
        y_train=y_train,
        x_query=x_test,
        kernel_variance=k_var,
        noise_variance=n_var,
    )

    # Test that predictions match
    assert torch.allclose(y_test, y_test2, atol=1e-3, rtol=1e-3)
    assert torch.allclose(y_test_covar, y_test_covar2, atol=1e-3, rtol=1e-3)

    # Exact GP mll
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp_exact.likelihood, gp_exact)
    gp_exact.train()
    with torch.no_grad(), gpytorch.settings.fast_computations(False, False, False):
        mll_val = mll(gp_exact(x_train), y_train) * len(x_train)  # they scale it down

    # Linear GP mll
    mll_val2 = linear_gp_mll(
        x_train=x_train,
        y_train=y_train,
        kernel_variance=k_var,
        noise_variance=n_var,
    )
    assert torch.isclose(
        mll_val,
        mll_val2,
    )
