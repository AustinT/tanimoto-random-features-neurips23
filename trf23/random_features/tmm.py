"""Random features for TMM kernel."""
from __future__ import annotations

import joblib
import numpy as np
from scipy import sparse
from tqdm.auto import tqdm

from trf23.fingerprint_features import FP_Dict


class TMM_Dict_Featurizer:
    """Random featurizer for TMM using i.i.d. entries of Ξ."""

    def __init__(
        self,
        max_fp_dim: int,
        num_features: int,
        rng,
        hash_mod: int = 4096,
        distribution="Rademacher",
        use_tqdm: bool = True,
        n_jobs=1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.max_fp_dim = max_fp_dim
        self.num_features = num_features
        self.hash_mod = hash_mod
        self.distribution = distribution
        self.rng = rng
        self.use_tqdm = use_tqdm
        self.n_jobs = n_jobs
        self._init_random_features()

    def _init_random_features(self):
        """Initialize all variances required to do consistent weighted sampling."""
        rand_size = (
            self.num_features,
            self.max_fp_dim,
        )

        # Regular CWS variables
        self._r = -np.log(self.rng.random(size=rand_size) * self.rng.random(size=rand_size))
        self._c = -np.log(self.rng.random(size=rand_size) * self.rng.random(size=rand_size))
        self._beta = self.rng.random(size=rand_size)

        xi_shape = (self.num_features, self.hash_mod)
        if self.distribution == "Rademacher":
            bin_samples = self.rng.integers(
                low=0,
                high=2,
                size=xi_shape,
            )
            self._xi = 2.0 * bin_samples - 1.0
        elif self.distribution == "Gaussian":
            self._xi = self.rng.normal(size=xi_shape)
        else:
            raise NotImplementedError()

    def __call__(self, x: list[FP_Dict]) -> np.ndarray:
        # NOTE: it is assumed that the fingerprints are already reduced to the max_fp_dim

        # Create output array
        out = list()

        def _create_features(fp):
            # Convert fingerprints to array
            fp_bits = np.array(list(fp.keys()))
            fp_vals = np.array([float(fp[k]) for k in fp_bits])

            # Get current CWS values
            r = self._r[:, fp_bits]
            c = self._c[:, fp_bits]
            beta = self._beta[:, fp_bits]

            # Calculate CWS values
            t = np.floor(np.log(fp_vals) / r + beta)
            ln_y = r * (t - beta)
            ln_a = np.log(c) - ln_y - r

            # Find argmin
            a_argmin_sparse = np.argmin(ln_a, axis=1)  # gives index in fp_bits
            a_argmin = [int(fp_bits[x]) for x in a_argmin_sparse]  # gives index in range(0, max_bits)

            # Create hashes as single integers
            hash_tuples = [
                (int(a_argmin[feat_idx]), int(t[feat_idx][a_argmin_sparse[feat_idx]]))
                for feat_idx in range(self.num_features)
            ]
            single_hashes = [hash(t) % self.hash_mod for t in hash_tuples]

            # Use hashes to index features
            return [float(self._xi[feat_idx, h]) for feat_idx, h in enumerate(single_hashes)]

        if self.use_tqdm:
            iterator = tqdm(x, desc="Making TMM features.")
        else:
            iterator = x
        out = joblib.Parallel(n_jobs=self.n_jobs)(joblib.delayed(_create_features)(fp) for fp in iterator)
        out_arr = np.asarray(out)
        return out_arr / np.sqrt(self.num_features)


class TMM_ArrFeaturizer(TMM_Dict_Featurizer):
    """Featurizer which accepts an array."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Convert to list of dicts
        x_dok = sparse.dok_array(x)
        fp_dicts: list[FP_Dict] = [dict() for _ in range(x.shape[0])]
        for k, v in x_dok.items():
            fp_dicts[k[0]][k[1]] = v
        return super().__call__(fp_dicts)
