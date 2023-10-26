from pathlib import Path

MAX_THREADS = 4
NUM_TRIALS = 3

for target in [
    "ESR2",
    "F2",
    "KIT",
    "PARP1",
    "PGR",
]:
    for M in [5000]:
        for count_fps in [True, False]:
            out_path = Path("results") / "regression" / f"M{M}" / f"count{count_fps}" / target
            out_path.mkdir(parents=True, exist_ok=True)
            for kernel in ["T_MM", "T_DP"]:
                for trial in range(NUM_TRIALS):
                    trial += 1  # start from 1
                    base_command = (
                        f"OMP_NUM_THREADS={MAX_THREADS} OPENBLAS_NUM_THREADS={MAX_THREADS} MKL_NUM_THREADS={MAX_THREADS} "
                        f"VECLIB_MAXIMUM_THREADS={MAX_THREADS} NUMEXPR_NUM_THREADS={MAX_THREADS} "
                        "CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:$PYTHONPATH python "
                        "experiment_scripts/approx_gp_regression.py "
                        f"--seed={trial} --dataset=dockstring --target={target} --fp_dim=1024 "
                        f"--kernel={kernel} --num_exact_fit=5_000 "
                    )
                    if not count_fps:
                        base_command += "--binary_fps "

                    def _get_command_with_output_files(expt_name):
                        expt_name = f"{expt_name}-{kernel}-{trial}"
                        return base_command + (
                            f"--logfile={out_path}/{expt_name}.log --output_json={out_path}/{expt_name}.json "
                        )

                    # Random subset of size M
                    print(_get_command_with_output_files("rsgp") + f"--eval_rsgp --rsgp_subset_sizes {M}")

                    # SVGP with M inducing points, set by k-means
                    svgp_batch_size = 2 * M
                    svgp_pretrain_num_steps = 220000 // svgp_batch_size  # approx size of dockstring dataset
                    print(
                        _get_command_with_output_files("svgp") + f"--fit_svgp --svgp_num_inducing_points {M} "
                        f"--svgp_pretrain_batch_size={svgp_batch_size} "
                        f"--svgp_pretrain_num_steps={svgp_pretrain_num_steps} "
                        f"--svgp_pretrain_eval_interval={svgp_pretrain_num_steps} "
                        "--svgp_pretrain_lr=1e-1 "
                        "--svgp_num_steps=0 "
                    )

                    # RFGP with M random features
                    standard_rfgp_args = f"--fit_rfgp --num_random_features={M} "
                    bias_correction_list = ["none"]
                    if kernel == "T_DP":
                        for bc in ["none", "normalize", "sketch_error"]:
                            print(
                                _get_command_with_output_files(f"rfgp-{bc}")
                                + standard_rfgp_args
                                + f"--tdp_bias_correction={bc} "
                            )
                    elif kernel == "T_MM":
                        for dist in ["Rademacher", "Gaussian"]:
                            print(
                                _get_command_with_output_files(f"rfgp-{dist}")
                                + standard_rfgp_args
                                + f"--tmm_distribution={dist} --num_jobs={MAX_THREADS} "
                            )
