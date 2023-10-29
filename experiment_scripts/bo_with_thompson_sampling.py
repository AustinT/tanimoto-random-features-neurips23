"""
Script to run BO with exact and approximate Thompson sampling.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import time

import numpy as np

from trf23 import fingerprint_features as ff
from trf23 import tanimoto_functions as tf
from trf23.datasets import dockstring
from trf23.random_features.tdp import Default_TDP_Featurizer
from trf23.random_features.tmm import TMM_ArrFeaturizer

logger = logging.getLogger(__name__)

NUM_FP_BITS = 1024


def main():
    # Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_size", type=int, required=True, help="Number of molecules to subsample for dataset.")
    parser.add_argument(
        "--target",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--num_random_features",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--num_jobs",
        type=int,
        default=-1,
    )
    args = parser.parse_args()

    # Randomness
    rng = np.random.default_rng(args.seed)

    # Load dockstring data
    smiles_train, smiles_test, y_train, y_test = dockstring.get_train_test_smiles(args.target)
    all_smiles = smiles_train + smiles_test
    all_y = np.concatenate([y_train, y_test])
    del smiles_train, smiles_test, y_train, y_test

    # Subsample dataset
    chosen_idxs = rng.choice(len(all_smiles), size=args.dataset_size, replace=False)
    all_smiles = [all_smiles[i] for i in chosen_idxs]
    all_y = all_y[chosen_idxs]

    # Make fingerprints
    fps = ff.fp_dicts_to_arr(ff.smiles_to_fp_dicts(all_smiles, radius=1), nbits=NUM_FP_BITS, binarize=False)
    fps_to_use = dict(TMM=fps, TDP=np.sqrt(fps))
    fps_to_use["TDP"] = fps_to_use["TDP"] / np.linalg.norm(fps_to_use["TDP"], axis=1, keepdims=True)
    del fps

    # Draw random samples (baseline)
    t_start = time.monotonic()
    random_samples = rng.uniform(size=(args.batch_size, len(all_smiles)))
    t_end = time.monotonic()
    thompson_samples = [
        dict(
            method="random",
            samples=random_samples,
            time=t_end - t_start,
        )
    ]

    # Exact and TMM and TDP samples
    tmm_featurizer = TMM_ArrFeaturizer(
        num_features=args.num_random_features, max_fp_dim=NUM_FP_BITS, rng=rng, use_tqdm=True, n_jobs=args.num_jobs
    )
    tdp_featurizer = Default_TDP_Featurizer(
        input_dim=NUM_FP_BITS,
        num_rf=args.num_random_features,
        rng=rng,
        bias_correction="none",
        min_input_square_norm=np.min(np.linalg.norm(fps_to_use["TDP"], axis=1)),
    )
    for k_name, k_func, k_feat in [
        ("TMM", tf.batch_tmm_sim_np, tmm_featurizer),
        ("TDP", tf.batch_tdp_sim_np, tdp_featurizer),
    ]:
        # Exact sample from GP
        t_start = time.monotonic()
        kernel_matix = k_func(fps_to_use[k_name], fps_to_use[k_name])
        samples = rng.multivariate_normal(
            mean=np.zeros(len(all_smiles)),
            cov=kernel_matix + np.eye(len(all_smiles)) * 1e-4,
            size=args.batch_size,
        )
        thompson_samples.append(
            dict(
                method=k_name + "_exact",
                samples=samples,
                time=time.monotonic() - t_start,
            )
        )

        # Approximate random feature samples
        t_start = time.monotonic()
        features = k_feat(fps_to_use[k_name])
        samples = rng.normal(size=(args.batch_size, args.num_random_features)) @ features.T
        thompson_samples.append(
            dict(
                method=k_name + "_rf",
                samples=samples,
                time=time.monotonic() - t_start,
            )
        )

    # Actually choose the best molecules
    output = []
    for d in thompson_samples:
        indices = select_batch_from_thompson_samples(d["samples"], batch_size=args.batch_size, forbidden_indices=set())
        output.append(
            dict(
                method=d["method"],
                time=d["time"],
                y_values=all_y[indices].tolist(),
            )
        )

    # Write to json file
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)


def select_batch_from_thompson_samples(
    samples: np.ndarray,
    batch_size: int,
    forbidden_indices: set[int],
) -> list[int]:
    """Samples has shape (n_samples, n_points)."""
    assert samples.ndim == 2
    samples_argsort = np.argsort(-samples, axis=1)
    out_set: set[int] = set()
    current_rank = 0
    while len(out_set) < batch_size:
        logger.debug(f"\tTS: Adding rank #{current_rank+1} samples.")
        counter = collections.Counter(list(samples_argsort[:, current_rank]))
        logger.log(level=5, msg=str(counter.most_common()))
        for i, _ in counter.most_common():
            if i not in forbidden_indices and len(out_set) < batch_size:
                out_set.add(i)
        logger.debug(f"Currently found {len(out_set)}/{batch_size} batch elements.")
        current_rank += 1
    out_list = list(out_set)
    assert len(out_list) == batch_size
    return out_list


if __name__ == "__main__":
    main()
