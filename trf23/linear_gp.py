"""
Code for GPs with linear kernels. Used to implement GPs with random features.
"""

from __future__ import annotations

import numpy as np
import torch


def _bayesian_linear_model_matrices(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    kernel_variance: torch.Tensor,
    noise_variance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return test-data independent matrices for a Bayesian linear model's predictions,
    *without* ever forming a NxN matrix (at most NxD, where N is the number of training points and D is dimension).

    The matrices are: (using notation of v = kernel variance, n = noise_variance)

    1. A = vX^T (vXX^T + nI)^{-1} y_train. Shape = D x N_train. X_test @ A = posterior mean.
    2. B = vX^T (vXX^T + nI)^{-1} @ X. Shape = D x D. v * (X_test @ X_test.T - X_test @ B @ X_test.T) = posterior covar.

    Overall it uses the Woodbury identity to avoid inverting the NxN matrix,
    and performs matrix multiplication in the right order to never form a NxN matrix.
    """

    # Precompare v / n (signal to noise ratio) since it is used a lot
    snr = kernel_variance / noise_variance
    snr2 = snr**2

    # DxD matrix which appears inside Woodbury identity and will be inverted
    inner_dxd_matrix = torch.eye(x_train.shape[-1]).to(x_train) + snr * x_train.T @ x_train  # call this matrix C

    # Precompute X^T @ X (used a lot)
    XtX = x_train.T @ x_train

    # Matrix A = v X^T (I/n - v/(n^2) X C^{-1} X^T) y_train (from Woodbury)
    #          = (v/n X^T y_train) - (v/n)^2 (X^T X) C^{-1} (X^T y_train)  (bracketing avoids NxN matrices)
    A = snr * x_train.T @ y_train - snr2 * (XtX) @ torch.linalg.solve(inner_dxd_matrix, x_train.T @ y_train)

    # Matrix B = v X^T (I/n - v/(n^2) X C^{-1} X^T) X (from Woodbury)
    #          = v/n X^T X - (v/n)^2 (X^T X) C^{-1} (X^T X)  (bracketing avoids NxN matrices)
    B = snr * XtX - snr2 * (XtX) @ torch.linalg.solve(inner_dxd_matrix, XtX)

    return A, B


def batch_linear_gp_predict(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_query_list: list[torch.Tensor],
    kernel_variance: torch.Tensor,
    noise_variance: torch.Tensor,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return mean and covar for a batch of queries, *without noise*."""
    A, B = _bayesian_linear_model_matrices(
        x_train=x_train,
        y_train=y_train,
        kernel_variance=kernel_variance,
        noise_variance=noise_variance,
    )

    return [(x @ A, kernel_variance * (x @ x.T - x @ B @ x.T)) for x in x_query_list]


def linear_gp_predict(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_query: torch.Tensor,
    kernel_variance: torch.Tensor,
    noise_variance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Efficiently predict mean and covariance of *zero-mean* linear GP,
    treating it as a Bayesian linear model.

    Note: predictions are returned *without noise*.
    """
    return batch_linear_gp_predict(
        x_train=x_train,
        y_train=y_train,
        x_query_list=[x_query],
        kernel_variance=kernel_variance,
        noise_variance=noise_variance,
    )[0]


def linear_gp_mll(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    kernel_variance: torch.Tensor,
    noise_variance: torch.Tensor,
) -> torch.Tensor:
    """
    Efficiently compute marginal log likelihood for a linear GP (on train data),
    *without* forming NxN matrices.
    """

    # Precompare v / n (signal to noise ratio) since it is used a lot
    snr = kernel_variance / noise_variance

    # DxD matrix which appears inside Woodbury identity and will be inverted
    inner_dxd_matrix = torch.eye(x_train.shape[-1]).to(x_train) + snr * x_train.T @ x_train  # call this matrix C

    # Precompute X^T @ y (used a lot)
    xty = x_train.T @ y_train.unsqueeze(1)

    # Data fit term:
    # y_train.T @ (vXX^T + nI)^{-1} y_train
    # = y_train.T @ (I/n - v/(n^2) X C^{-1} X^T) y_train (from Woodbury)
    # = |y_train|^2/n - v/n^2 (y_train^T X) C^{-1} (X^T y_train) (bracketing avoids NxN matrices)
    data_fit = (
        torch.sum(y_train**2) / noise_variance
        - snr * xty.T @ torch.linalg.solve(inner_dxd_matrix, xty) / noise_variance
    ).squeeze()

    # Model complexity term:
    # log |vXX^T + nI|
    # = log(n ^ num_train) + log|C|
    model_complexity = (
        len(x_train) * torch.log(torch.as_tensor(noise_variance)) + torch.linalg.slogdet(inner_dxd_matrix)[1]
    )

    # Return overall log likelihood
    return -0.5 * (data_fit + model_complexity + len(y_train) * np.log(2 * np.pi))
