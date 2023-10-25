import math

import gpytorch
import numpy as np
import pytest
import sklearn.metrics
import torch
from gpytorch.kernels import ScaleKernel

from trf23 import fingerprint_features as ff
from trf23.tanimoto_gp import (
    CustomKernelSVGP,
    TanimotoKernelGP,
    TDP_Kernel,
    TMM_Kernel,
    batch_predict_mu_std_numpy,
    fit_exact_gp_hyperparameters_scipy,
    get_gp_hyperparameters,
)


def _test_toy_data(gp):
    test_x = torch.as_tensor([[1.0, 1.0, 1.0, 0.0]])
    gp.eval()
    with gpytorch.settings.fast_computations(False, False, False), torch.no_grad():
        output = gp(test_x)
        output_mean = output.mean.detach().numpy()
        output_std = output.stddev.detach().numpy()

    assert np.allclose(output_mean, -0.5, atol=1e-3)
    assert np.allclose(output_std, 0.866, atol=1e-3)


@pytest.mark.parametrize(
    "kernel",
    [
        TDP_Kernel(),
        TMM_Kernel(),
    ],
)
def test_gp_creation_toy_data(kernel: gpytorch.kernels.Kernel):
    """Create a GP and run some tests in a toy setting."""

    train_x = torch.as_tensor([[1.0, 0.0, 1.0, 1.0]])
    train_y = torch.as_tensor([-1.0])

    # Exact GP (no noise)
    exact_gp = TanimotoKernelGP(
        train_x=train_x, train_y=train_y, kernel=ScaleKernel(kernel), mean_obj=gpytorch.means.ZeroMean()
    )
    exact_gp.likelihood.noise = 1e-4  # essentially zero noise

    # SVGP
    svgp = CustomKernelSVGP(kernel_obj=ScaleKernel(kernel), inducing_points=train_x, mean_obj=gpytorch.means.ZeroMean())
    svgp.set_inducing_points_requires_grad(False)
    with torch.no_grad():
        svgp(train_x)  # need to call it to initialize variational parameters

        # Set pseudo-observations to be train y (so it will behave the same as an exact GP)
        svgp.variational_strategy._variational_distribution.variational_mean.data = train_y.clone()

        # Set variational cov to be small to represent low noise
        svgp.variational_strategy._variational_distribution.chol_variational_covar.data[:] = 1e-4

    # Test all GPs
    for gp in [exact_gp, svgp]:
        # Get kernel and set outputscale (tests get_kernel)
        k = gp.covar_module
        k.outputscale = 1.0
        gp.train()  # type: ignore  # mypy doesn't correctly infer type of gp here

        # Run tests
        _test_toy_data(gp)


def test_end_to_end_regression(smiles_and_logp):
    """
    Make a Minmax fingerprint GP, fit to a dataset, make predictions.

    Test that everything runs and that the end performance is decent.
    """

    # Make features
    smiles_list, y = smiles_and_logp
    fp_arr = ff.fp_dicts_to_arr(ff.smiles_to_fp_dicts(smiles_list, radius=2), nbits=1024, binarize=False)

    # Prepare train/test splits
    num_train = 75
    X_train = fp_arr[:num_train]
    X_test = fp_arr[num_train:]
    y_train = y[:num_train]
    y_test = y[num_train:]

    # Fit GP
    gp = TanimotoKernelGP(
        train_x=torch.as_tensor(X_train),
        train_y=torch.as_tensor(y_train),
        kernel="T_MM",
        mean_obj=gpytorch.means.ZeroMean(),
    )
    fit_exact_gp_hyperparameters_scipy(gp)

    # Test 1: test fit quality.
    # The GP should fit the data well, so the outputscale should be much larger than the noise
    hparams = get_gp_hyperparameters(gp)
    assert gp.covar_module.outputscale.item() > 10 * gp.likelihood.noise.item()
    assert math.isclose(hparams["outputscale"], gp.covar_module.outputscale.item())

    # Make predictions using batch predict function
    y_pred, y_pred_std = batch_predict_mu_std_numpy(gp, X_test)

    # Test 2: predictions should be better than the mean (r2 > 0)
    assert sklearn.metrics.r2_score(y_true=np.asarray(y_test), y_pred=y_pred) > 0

    # Test 3: pred std should be lower than GP output scale, but not zero
    assert np.all(y_pred_std < np.sqrt(gp.covar_module.outputscale.item())) and np.all(y_pred_std > 0)

    # Test 4: if include_std = False, then mean should match, but std should be 0
    y_pred2, y_pred_std2 = batch_predict_mu_std_numpy(gp, X_test, include_std=False)
    assert np.allclose(y_pred, y_pred2)
    assert np.allclose(y_pred_std2, 0)
