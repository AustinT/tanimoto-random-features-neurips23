"""Analyzes error of TMM random features."""
import datetime
import json

import numpy as np
from tmm_random_feature_analysis import LAMBDA_ARR, NUM_RF_ARR, NUM_RF_TRIALS, get_fingerprints, get_parser

from trf23 import tanimoto_functions as tf
from trf23.random_features import tdp as rf
from trf23.random_features.error import max_eigenvalue_difference

R_LIST = list(range(1, 6))
P_LIST = np.linspace(-2, 2, 9).tolist()
NUM_PREFACTOR_RF = 10_000
SC_MULT_ARR = np.logspace(-2, 2, 15)


def main():
    # Initialize output
    output = dict(
        lambda_arr=LAMBDA_ARR.tolist(),
        num_rf_arr=NUM_RF_ARR,
        R_LIST=R_LIST,
        P_LIST=P_LIST,
    )

    # Get fingerprints
    args = get_parser().parse_args()
    fps = get_fingerprints(binarize=args.binary_fps)
    fps = np.sqrt(fps)  # both for count and binary

    # Normalize fingerprints
    fp_norms = np.linalg.norm(fps, axis=1)
    max_norm = np.max(fp_norms)
    fps /= max_norm
    min_norm = np.min(np.linalg.norm(fps, axis=1))
    min_norm2 = float(min_norm**2)
    output["normalization"] = dict(
        max_norm=float(max_norm),
        min_norm=float(min_norm),
    )

    # Compute exact kernel
    tdp_exact = tf.batch_tdp_sim_np(fps, fps)

    # Compute exact partial results
    prefactors = {r: rf.prefactor_exact(fps, fps, r=r) for r in R_LIST}
    poly_terms = {r: np.power(fps @ fps.T, r) for r in R_LIST}
    tdp_terms = {r: rf.tdp_power_series_term(fps, fps, r=r) for r in R_LIST}

    # Repeat studies for different trials and different numbers of random features
    mses = dict()
    eig_diffs = dict()
    for trial in range(NUM_RF_TRIALS):
        rng = np.random.default_rng(trial)
        for n_rf in NUM_RF_ARR:

            def _log_error(true, approx, key):
                mse = np.mean((true - approx) ** 2)
                eig = max_eigenvalue_difference(
                    K_exact=true,
                    K_approx=approx,
                    lambda_arr=LAMBDA_ARR,
                )
                mses.setdefault(key, dict()).setdefault(n_rf, []).append(mse)
                eig_diffs.setdefault(key, dict()).setdefault(n_rf, []).append(eig)

            for r in R_LIST:
                print(f"Trial {trial+1}", f"M={n_rf}", f"r={r}", datetime.datetime.now())

                # Measurement 1: error of prefactor random features
                s_opt = rf._get_optimal_s(r, min_norm2)
                c_opt = rf._get_optimal_c(r, min_norm2)
                featurizer = rf.PrefactorQMC_Featurizer(
                    r=r,
                    num_rf=n_rf,
                    s=s_opt,
                    c=c_opt,
                    rng=rng,
                )
                features = featurizer(fps)
                _log_error(true=prefactors[r], approx=features @ features.T, key=f"prefactor_{r}")

                # Measurement 2: error of polynomial random features
                featurizer = rf.TDP_PowerSeriesTermFeaturizer(
                    r=r,
                    num_rf=n_rf,
                    num_prefactor_rf=NUM_PREFACTOR_RF,
                    s=rf._get_optimal_s(r, min_norm2),
                    c=rf._get_optimal_c(r, min_norm2),
                    rng=rng,
                    input_dim=fps.shape[1],
                )
                features = featurizer.numerator_sketch.transform(fps)
                _log_error(true=poly_terms[r], approx=features @ features.T, key=f"poly_{r}")

                # Measurement 3: total error of features for this term.
                # (uses same featurizer as above)
                features = featurizer(fps)
                _log_error(true=tdp_terms[r], approx=features @ features.T, key=f"tdp-term_{r}")

                # Measurement 4: error of entire TDP kernel with different feature allocations
                for bias_correction in ["none", "normalize", "sketch_error"]:
                    if r == 1 and bias_correction == "sketch_error":
                        continue  # nothing can be done
                    for allocation_p in P_LIST:
                        featurizer = rf.Default_TDP_Featurizer(
                            input_dim=fps.shape[1],
                            num_rf=n_rf,
                            max_R=r,
                            r_allocation_p=allocation_p,
                            num_gamma=NUM_PREFACTOR_RF,
                            min_input_square_norm=min_norm2,
                            rng=rng,
                            bias_correction=bias_correction,
                        )
                        features = featurizer(fps)
                        _log_error(
                            true=tdp_exact,
                            approx=features @ features.T,
                            key=f"tdp_{bias_correction}_p{allocation_p}_R{r}",
                        )

                # Measurement 5: error of prefactor sketch with different s, c
                if n_rf == 10_000:
                    print("Doing S, C study")
                    sc_mses = []
                    for s in s_opt * SC_MULT_ARR:
                        sc_mses.append([])
                        for c in c_opt * SC_MULT_ARR:
                            featurizer = rf.PrefactorQMC_Featurizer(
                                r=r,
                                num_rf=n_rf,
                                s=s,
                                c=c,
                                rng=rng,
                            )
                            features = featurizer(fps)
                            sc_mses[-1].append(np.mean((prefactors[r] - features @ features.T) ** 2))
                    output.setdefault(f"prefactor_r{r}_M{n_rf}_sc_mses", []).append(sc_mses)
                    output["SC_MULT_ARR"] = SC_MULT_ARR.tolist()

    # Save output
    output["mses"] = mses
    output["eig_diffs"] = eig_diffs
    print("Saving outputs...", datetime.datetime.now())
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
