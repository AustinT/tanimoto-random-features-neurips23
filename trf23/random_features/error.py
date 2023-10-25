from __future__ import annotations

import numpy as np


def max_eigenvalue_difference(*, K_exact, K_approx, lambda_arr):
    """
    Compute solution to regularized eigenvalue problem:

    max |x^T (K_approx - K) x|

    subject to: x^T(K+lambda I)x <1

    This is equivalent to the largest singular value of:

    (K + lambda I)^{-.5} (K_approx - K) (K+lambda I)^{-0.5}
    """

    # Compute SVD of K_exact (exploiting that it is a hermitian matrix)
    U, sigma, V = np.linalg.svd(K_exact, hermitian=True)

    output: list[float] = []
    for lam in lambda_arr:
        K_plus_lambdaI_sqrt = U @ np.diag((sigma + lam) ** -0.5) @ V
        M = K_plus_lambdaI_sqrt @ (K_approx - K_exact) @ K_plus_lambdaI_sqrt
        sing_vals = np.linalg.svd(M, compute_uv=False)
        assert np.all(sing_vals > 0)
        output.append(float(np.max(sing_vals)))
    return output
