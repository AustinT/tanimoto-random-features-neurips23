"""Analyzes error of TMM random features."""
import argparse
import json

import numpy as np

from trf23 import fingerprint_features as ff
from trf23 import tanimoto_functions as tf
from trf23.datasets.guacamol import get_guacamol_smiles
from trf23.random_features.error import max_eigenvalue_difference
from trf23.random_features.tmm import TMM_ArrFeaturizer

NUM_RF_ARR = [int(n) for n in np.logspace(2, 5, 7)]
LAMBDA_ARR = np.logspace(-4, 1, 6)
NUM_RF_TRIALS = 5
NUM_VAR_MEASUREMENTS_PER_BUCKET = 10


def get_fingerprints(binarize: bool) -> np.ndarray:
    smiles = get_guacamol_smiles(1000)
    fp_dicts = ff.smiles_to_fp_dicts(smiles, radius=2)
    return ff.fp_dicts_to_arr(fp_dicts, nbits=1024, binarize=binarize)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--binary_fps", action="store_true")
    parser.add_argument("--num_jobs", type=int, default=-1)
    return parser


def main():
    args = get_parser().parse_args()
    fps = get_fingerprints(binarize=args.binary_fps)

    # Compute exact kernel
    tmm_exact = tf.batch_tmm_sim_np(fps, fps)

    # Which input pairs to choose for variance calculation?
    # Choose pseudo-evenly spaced pairs
    interval_size = 0.05
    input_idxs1 = []
    input_idxs2 = []
    rng = np.random.default_rng(105)
    for i_start in np.arange(0, 1, interval_size):
        i_end = i_start + interval_size
        in_interval = np.where((i_start < tmm_exact) & (tmm_exact <= i_end))
        if len(in_interval[0]) > NUM_VAR_MEASUREMENTS_PER_BUCKET:
            interval_idxs = rng.choice(len(in_interval[0]), size=NUM_VAR_MEASUREMENTS_PER_BUCKET, replace=False)
            in_interval = (in_interval[0][interval_idxs], in_interval[1][interval_idxs])
        input_idxs1.extend(in_interval[0].tolist())
        input_idxs2.extend(in_interval[1].tolist())

    # Compute approximate kernel with various numbers of random features.
    # Use fact that features are i.i.d. to only compute one set of features
    mses = dict()
    eig_diffs = dict()
    tmm_variances = dict()
    for trial in range(NUM_RF_TRIALS):
        for dist in ["Rademacher", "Gaussian"]:
            print(trial, dist)

            # Make the features
            featurizer = TMM_ArrFeaturizer(
                max_fp_dim=fps.shape[1],
                num_features=max(NUM_RF_ARR),
                rng=np.random.default_rng(trial),
                distribution=dist,
                use_tqdm=True,
                n_jobs=args.num_jobs,
            )
            all_features = featurizer(fps)

            # Compute errors for different feature lengths
            for n_rf in NUM_RF_ARR:
                features = all_features[:, :n_rf] * np.sqrt(featurizer.num_features / n_rf)
                tmm_approx = features @ features.T
                mse = np.mean((tmm_exact - tmm_approx) ** 2)
                eig_diff = max_eigenvalue_difference(
                    K_exact=tmm_exact,
                    K_approx=tmm_approx,
                    lambda_arr=LAMBDA_ARR,
                )
                mses.setdefault(dist, dict()).setdefault(n_rf, []).append(mse)
                eig_diffs.setdefault(dist, dict()).setdefault(n_rf, []).append(eig_diff)

            # For the first trial, compute variance for different input pairs
            if trial == 0:
                tmm_variances[dist] = dict(tmm=[], var=[])
                for i, j in zip(input_idxs1, input_idxs2):
                    estimates = all_features[i, :] * all_features[j, :] * all_features.shape[1]
                    tmm_variances[dist]["tmm"].append(float(tmm_exact[i, j]))
                    tmm_variances[dist]["var"].append(float(np.var(estimates)))

    # Save results
    res_dict = dict(
        mses=mses,
        eig_diffs=eig_diffs,
        tmm_variances=tmm_variances,
        lambda_arr=LAMBDA_ARR.tolist(),
        num_rf_arr=NUM_RF_ARR,
    )
    with open(args.output_json, "w") as f:
        json.dump(res_dict, f, indent=2)


if __name__ == "__main__":
    main()
