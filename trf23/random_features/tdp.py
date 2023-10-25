"""Random features for TDP kernel."""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Optional

import numpy as np
from scipy import stats
from sklearn.kernel_approximation import PolynomialCountSketch, fft, ifft


def gamma_dist(s: float, c: float) -> stats.gamma:
    return stats.gamma(a=s, scale=1 / c)


def draw_gamma_qmc_samples(num_rf: int, s: float, c: float, rng: np.random.Generator) -> np.ndarray:
    gamma_offset = rng.random()
    gamma_cdf_vals = np.mod(gamma_offset + np.linspace(0, 1, num_rf), 1.0)
    return gamma_dist(s=s, c=c).ppf(gamma_cdf_vals)


def tanimoto_power_series_denominator_rf(
    x: np.ndarray,
    gamma_samples: np.ndarray,
    r: int,
    c: float = 1.0,
    s: Optional[float] = None,
    allow_r_less_than_s: bool = False,
):
    """Random feature function for (|x|^2+|y|^2)^{-r})."""

    # Fill in default for s
    if s is None:
        s = float(r)
    assert allow_r_less_than_s or r >= s

    # Check shapes
    assert x.ndim == 2
    assert gamma_samples.ndim == 1
    x_norm2 = np.sum(x**2, axis=-1, keepdims=True)  # to allow broadcasting with z

    # Important: calculate *log* of random features for numerical stability
    term1 = (c / 2.0 - x_norm2) * gamma_samples
    term2 = (r - s) / 2.0 * np.log(gamma_samples)
    term3 = 0.5 * (-s * math.log(c) + math.lgamma(s) - math.lgamma(r))
    log_rf = term1 + term2 + term3
    rf_not_normalized = np.exp(log_rf)
    return rf_not_normalized / math.sqrt(rf_not_normalized.shape[-1])


def combine_count_sketches(sketch1: np.ndarray, sketch2: np.ndarray) -> np.ndarray:
    # NOTE: this code bit was minimally adapted from the sklearn source code
    fft_prod = fft(sketch1, axis=1, overwrite_x=True) * fft(sketch2, axis=1, overwrite_x=True)
    return np.real(ifft(fft_prod, overwrite_x=True))


class TDP_PowerSeriesTermFeaturizer:
    """
    Random featurizer for a single term in the TDP power series,
    (<x,y> / (|x|^2 + |y|^2))^r.

    This implementation uses CountSketch for the polynomial sketch,
    implemented in scikit-learn.
    """

    def __init__(
        self,
        *,
        num_rf: int,
        gamma_samples: np.ndarray,
        input_dim: int,
        c: float,
        s: float,
        r: int,
        rng,
    ):
        super().__init__()

        self.r = r
        self.s = s
        self.c = c
        self._gamma_samples = gamma_samples

        random_state = np.random.RandomState(rng.bit_generator)
        self.numerator_sketch = PolynomialCountSketch(degree=self.r, n_components=num_rf, random_state=random_state)
        self.numerator_sketch.fit(np.zeros((1, input_dim)))  # just initialize random variables
        self.denominator_sketch = PolynomialCountSketch(degree=1, n_components=num_rf, random_state=random_state)
        self.denominator_sketch.fit(np.zeros((1, len(gamma_samples))))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Sketch denominator
        phi = tanimoto_power_series_denominator_rf(x, self._gamma_samples, self.r, s=self.s, c=self.c)

        # Sketch numerator
        num_rf_sketch = self.numerator_sketch.transform(x)
        den_rf_sketch = self.denominator_sketch.transform(phi)

        # Combine sketches with Fourier transform
        return combine_count_sketches(num_rf_sketch, den_rf_sketch)


class TDP_PowerSeriesFeaturizer:
    def __init__(
        self,
        *,
        input_dim: int,
        r_to_num_rf: dict[int, int],
        r_to_num_gamma: dict[int, int],
        r_to_s: dict[int, float],
        r_to_c: dict[int, float],
        rng: np.random.Generator,
        bias_correction: str = "none",
    ):
        super().__init__()

        # Some argument checks
        # Check 1: list of r values should be 1, 2, ..., R (i.e. a contiguous list of integers)
        self.R = max(r_to_num_rf.keys())
        assert set(r_to_num_rf.keys()) == set(range(1, self.R + 1))

        # Check 2: r values in other input dictionaries should all match
        assert set(r_to_num_rf.keys()) == set(r_to_num_gamma.keys())
        assert set(r_to_num_rf.keys()) == set(r_to_s.keys())
        assert set(r_to_num_rf.keys()) == set(r_to_c.keys())

        # Create random featurizers for each term
        self.term_featurizers = []
        for r in self.r_list:
            gamma_samples = draw_gamma_qmc_samples(
                num_rf=r_to_num_gamma[r],
                s=r_to_s[r],
                c=r_to_c[r],
                rng=rng,
            )
            featurizer_r = TDP_PowerSeriesTermFeaturizer(
                num_rf=r_to_num_rf[r],
                gamma_samples=gamma_samples,
                input_dim=input_dim,
                c=r_to_c[r],
                s=r_to_s[r],
                r=r,
                rng=rng,
            )
            self.term_featurizers.append(featurizer_r)

        # Bias correction
        if bias_correction == "none":
            self.combine_sketch_terms = self.concatenate_terms
        elif bias_correction == "normalize":
            self.combine_sketch_terms = self.concatenate_terms_with_normalization
        elif bias_correction == "sketch_error":
            self.combine_sketch_terms = self.concatenate_terms_with_error_sketch

            # Sketch 1: the terms r=1 to R-1
            self._truncation_error_sketch1 = PolynomialCountSketch(
                degree=1, n_components=r_to_num_rf[self.R], random_state=np.random.RandomState(rng.bit_generator)
            )
            self._truncation_error_sketch1.fit(np.zeros((1, 1 + sum(r_to_num_rf[r] for r in self.r_list[:-1]))))

            # Sketch 2 (for the final term)
            self._truncation_error_sketch2 = PolynomialCountSketch(
                degree=1, n_components=r_to_num_rf[self.R], random_state=np.random.RandomState(rng.bit_generator)
            )
            self._truncation_error_sketch2.fit(np.zeros((1, r_to_num_rf[self.R])))
        else:
            raise ValueError(bias_correction)

    @property
    def r_list(self) -> Sequence[int]:
        return range(1, self.R + 1)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Compute each term
        r_to_term = {r: featurizer(x) for r, featurizer in zip(self.r_list, self.term_featurizers)}

        # Combine terms
        return self.combine_sketch_terms(r_to_term)

    def concatenate_terms(self, r_to_term: dict[int, np.ndarray]) -> np.ndarray:
        """Standard way to combine random feature estimates for each term"""
        return np.concatenate([r_to_term[r] for r in self.r_list], axis=-1)

    def concatenate_terms_with_normalization(self, r_to_term: dict[int, np.ndarray]) -> np.ndarray:
        """Concatenate terms and then normalize them"""
        out = self.concatenate_terms(r_to_term=r_to_term)
        return out / np.linalg.norm(out, axis=-1, keepdims=True)

    def concatenate_terms_with_error_sketch(self, r_to_term: dict[int, np.ndarray]) -> np.ndarray:
        """Concatenate the terms with an extra error sketch"""

        # Get all sketches
        first_sketches = np.concatenate([r_to_term[r] for r in self.r_list[:-1]], axis=-1)
        first_sketches_with_one = np.concatenate([np.ones((r_to_term[1].shape[0], 1)), first_sketches], axis=-1)
        final_term_sketch = r_to_term[self.r_list[-1]]

        # Use tensor product to estimate truncation error
        sketch1 = self._truncation_error_sketch1.transform(first_sketches_with_one)
        sketch2 = self._truncation_error_sketch2.transform(final_term_sketch)
        error_sketch = combine_count_sketches(sketch1, sketch2)

        # Concatenate and return
        return np.concatenate([first_sketches, error_sketch], axis=-1)


class Default_rs_TDPFeaturizer(TDP_PowerSeriesFeaturizer):
    """A power series featurizer for the Tanimoto kernel with r,s values based on input norms."""

    def __init__(
        self,
        *,
        input_dim: int,
        r_to_num_rf: dict[int, int],
        num_gamma: int,
        min_input_square_norm: float,
        **kwargs,
    ):
        # Compute r_to_* dicts
        r_to_s = {r: r * min_input_square_norm for r in r_to_num_rf}
        r_to_c = {r: 2 * (min_input_square_norm**2) for r in r_to_num_rf}
        r_to_num_gamma = {r: num_gamma for r in r_to_num_rf}

        super().__init__(
            input_dim=input_dim,
            r_to_num_rf=r_to_num_rf,
            r_to_num_gamma=r_to_num_gamma,
            r_to_s=r_to_s,
            r_to_c=r_to_c,
            **kwargs,
        )


class Default_TDP_Featurizer(Default_rs_TDPFeaturizer):
    def __init__(
        self,
        *,
        input_dim: int,
        num_rf: int,
        max_R: int = 4,
        **kwargs,
    ):
        # Set some default kwargs
        kwargs.setdefault("num_gamma", 10_000)

        # r to num rf
        r_array = np.arange(1, max_R + 1)
        rf_fracs = 1 / r_array
        rf_fracs = rf_fracs / rf_fracs.sum()
        r_to_num_rf = {r: int(num_rf * rf_frac) for r, rf_frac in zip(r_array, rf_fracs)}
        total_rf = sum(r_to_num_rf.values())
        if sum(r_to_num_rf.values()) < num_rf:
            r_to_num_rf[1] += num_rf - total_rf  # assign remainder to r=1

        super().__init__(
            input_dim=input_dim,
            r_to_num_rf=r_to_num_rf,
            **kwargs,
        )
