"""Code for various Tanimoto similarity functions (T_MM and T_DP)."""

from __future__ import annotations

import numpy as np
import torch


def _tdp_numerator_denominator(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Compute the numerator and denominator of TDP."""
    assert x1.ndim >= 2 and x2.ndim >= 2
    dot_prod = torch.matmul(x1, torch.transpose(x2, -1, -2))
    x1_sum = torch.sum(x1**2, dim=-1, keepdims=True)
    x2_sum = torch.sum(x2**2, dim=-1, keepdims=True)
    return dot_prod, x1_sum + torch.transpose(x2_sum, -1, -2) - dot_prod


def batch_tdp_sim(x1: torch.Tensor, x2: torch.Tensor, eps=1e-10) -> torch.Tensor:
    """
    Tanimoto (TDP) similarity between two batched tensors, across last 2 dimensions.

    eps argument ensures numerical stability if all zero tensors are added.
    """
    numerator, denominator = _tdp_numerator_denominator(x1, x2)
    return (numerator + eps) / (denominator + eps)


def batch_tdp_sim_np(x1: np.ndarray, x2: np.ndarray, eps=1e-10) -> np.ndarray:
    return batch_tdp_sim(torch.from_numpy(x1), torch.from_numpy(x2), eps=eps).numpy()


def _minsum_maxsum(x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given x1 (NxD) and x2 (MxD), this method returns two NxM tensors A, B
    where:

    A[i,j] = 2*sum(minimum(x1[i], x2[j]))

    B[i,j] = 2*sum(maximum(x1[i], x2[j]))

    It follows that A / B (elementwise) equals the min-max kernel.
    """

    # Check dimension is correct
    assert x1.ndim == 2
    assert x2.ndim == 2
    assert x1.shape[1] == x2.shape[1]

    # Compute l1 norms
    x1_norm = torch.sum(x1, dim=-1, keepdim=True)
    x2_norm = torch.sum(x2, dim=-1, keepdim=True)
    norm_sum = x1_norm + x2_norm.T
    pairwise_dist = torch.cdist(x1, x2, p=1)

    return (norm_sum - pairwise_dist, norm_sum + pairwise_dist)


def batch_tmm_sim(x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Computes the min-max similarity between two non-negative matrices.
    """
    if x1.ndim == x2.ndim == 2:
        min_sum, max_sum = _minsum_maxsum(x1, x2)
        return (min_sum + eps) / (max_sum + eps)
    elif x1.ndim == x2.ndim == 3:
        # Batch mode
        assert x1.shape[0] == x2.shape[0]
        out = []
        for _x1, _x2 in zip(x1, x2):
            out.append(batch_tmm_sim(_x1, _x2, eps=eps))
        return torch.stack(out)
    else:
        raise NotImplementedError


def batch_tmm_sim_np(x1: np.ndarray, x2: np.ndarray, eps=1e-10) -> np.ndarray:
    return batch_tmm_sim(torch.from_numpy(x1), torch.from_numpy(x2), eps=eps).numpy()
