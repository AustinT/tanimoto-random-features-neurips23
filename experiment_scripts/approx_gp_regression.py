"""Script to run experiments for approximate GP regression."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pprint import pformat
from typing import Any

import gpytorch
import numpy as np
import sklearn.metrics
import torch
from scipy import stats
from sklearn.cluster import KMeans
from tqdm.auto import tqdm

from trf23 import fingerprint_features as ff
from trf23.datasets import dockstring
from trf23.linear_gp import batch_linear_gp_predict
from trf23.random_features.tdp import Default_TDP_Featurizer
from trf23.random_features.tmm import TMM_ArrFeaturizer
from trf23.tanimoto_gp import (
    CustomKernelSVGP,
    TanimotoKernelGP,
    batch_predict_mu_std_numpy,
    fit_exact_gp_hyperparameters_scipy,
    get_gp_hyperparameters,
)

logger = logging.getLogger(__name__)

GP_EVAL_BATCH_SIZE = 1024
RF_PREDICT_BATCH_SIZE = 16


def main():
    # Read argument, set up logging
    parser = get_parser()
    args = parser.parse_args()
    logging.basicConfig(
        filename=args.logfile,
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        filemode="w",
    )
    logger.setLevel(logging.DEBUG)
    logger.info("START OF SCRIPT")
    logger.info(args)
    output: dict[str, Any] = dict(times=dict(), metrics=dict())
    output["args"] = args.__dict__

    # Randomness
    rng = np.random.default_rng(args.seed)

    # Load data
    if args.dataset == "dockstring":
        logger.debug("Loading dockstring dataset...")
        smiles_train, smiles_test, y_train, y_test = dockstring.get_train_test_smiles(args.target)
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    # Make fingerprints
    t_start = time.monotonic()
    logger.info("Making fingerprints")
    fp_kwargs = dict(radius=1)
    fp_arr_kwargs = dict(nbits=args.fp_dim, binarize=args.binary_fps)
    fp_train = ff.fp_dicts_to_arr(ff.smiles_to_fp_dicts(smiles_train, **fp_kwargs), **fp_arr_kwargs)
    fp_test = ff.fp_dicts_to_arr(ff.smiles_to_fp_dicts(smiles_test, **fp_kwargs), **fp_arr_kwargs)
    logger.info("Done making fingerprints")
    output["times"]["make_fps"] = time.monotonic() - t_start
    output["data_shapes"] = dict(train=fp_train.shape, test=fp_test.shape)

    # If using TDP kernel, square root the features to keep interpretation similar
    if args.kernel == "T_DP":
        logger.debug("Square rooting features for TDP kernel")
        fp_train = np.sqrt(fp_train)
        fp_test = np.sqrt(fp_test)

    # Create exact GP on subset of data to determine hyperparameters
    gp_exact_subset_indices = rng.choice(len(fp_train), size=args.num_exact_fit, replace=False)
    exact_subset_gp = TanimotoKernelGP(
        train_x=torch.as_tensor(fp_train[gp_exact_subset_indices]),
        train_y=torch.as_tensor(y_train[gp_exact_subset_indices]),
        kernel=args.kernel,
        mean_obj=gpytorch.means.ConstantMean(),
    )

    # Initialize hyperparameters from known train labels
    exact_subset_gp.mean_module.initialize(constant=np.mean(y_train))
    y_train_var = np.var(y_train)
    exact_subset_gp.covar_module.initialize(outputscale=y_train_var)
    exact_subset_gp.likelihood.initialize(noise=y_train_var / 10.0)

    # Optimize hyperparameters with scipy
    logger.info("Fitting subset GP")
    t_start = time.monotonic()
    fit_exact_gp_hyperparameters_scipy(exact_subset_gp)
    output["times"]["fit_subset_gp"] = time.monotonic() - t_start
    gp_hparams = get_gp_hyperparameters(exact_subset_gp)
    output["gp_hparams"] = gp_hparams
    logger.info(f"Done fitting subset GP. Params: {pformat(gp_hparams)}")

    # Turn off gradients for exact GP hyperparameters: won't be needed anymore
    for p in exact_subset_gp.parameters():
        p.requires_grad_(False)

    # Possibly eval random subset GP
    if args.eval_rsgp:
        logger.info("evaluating RSGP")
        for i, subset_size in enumerate(args.rsgp_subset_sizes):
            logger.debug(f"Eval subset size {subset_size}")
            time_taken, metrics = _eval_subset_gp(
                reference_gp=exact_subset_gp,
                rng=rng,
                subset_size=subset_size,
                fp_train=fp_train,
                y_train=y_train,
                fp_test=fp_test,
                y_test=y_test,
            )

            # NOTE: "i" key ensures the same subset size can be tried multiple times
            key = f"{i};{subset_size}"
            output["times"].setdefault("eval_subset_gp", dict())[key] = time_taken
            output["metrics"].setdefault("eval_subset_gp", dict())[key] = metrics
            del metrics, time_taken, subset_size
    else:
        logger.info("NOT evaluating RSGP")

    # Possibly fit sparse GP
    if args.fit_svgp:
        logger.info("fitting SVGP")
        _fit_and_eval_svgp(
            reference_gp=exact_subset_gp,
            rng=rng,
            fp_train=fp_train,
            y_train=y_train,
            fp_test=fp_test,
            y_test=y_test,
            output=output,
            args=args,
        )
    else:
        logger.info("NOT fitting SVGP")

    # Possibly fit random feature GP
    if args.fit_rfgp:
        logger.info("fitting RF-GP")
        _fit_and_eval_rfgp(
            reference_gp=exact_subset_gp,
            rng=rng,
            fp_train=fp_train,
            y_train=y_train,
            fp_test=fp_test,
            y_test=y_test,
            output=output,
            args=args,
        )
    else:
        logger.info("NOT fitting RF-GP")

    # Write output and terminate
    logger.info("Writing output json...")
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("END OF SCRIPT")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # IO arguments
    parser.add_argument(
        "--logfile",
        type=str,
        default="tmp.log",
        help="file to log to.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Path to output JSON file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed to use.",
    )
    parser.add_argument(
        "--num_jobs",
        type=int,
        default=1,
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["dockstring"],
        help="Which dataset to use.",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target column to take from the CSV",
    )
    parser.add_argument(
        "--fp_dim",
        type=int,
        default=1024,
        help="Dimension of fingerprint to use.",
    )
    parser.add_argument(
        "--binary_fps",
        action="store_true",
        help="Flag to use binary FPs.",
    )

    # GP arguments
    parser.add_argument(
        "--kernel",
        type=str,
        required=True,
        choices=["T_DP", "T_MM"],
        help="Kernel to use.",
    )
    parser.add_argument(
        "--num_exact_fit",
        type=int,
        default=1000,
        help="Number of points to subsample to fit exact GP.",
    )

    # Random subset GP arguments
    parser.add_argument("--eval_rsgp", action="store_true", help="Flag to run random subset GP eval.")
    parser.add_argument(
        "--rsgp_subset_sizes",
        type=int,
        nargs="+",
        default=[],
        help="Size of random subset to use for random subset GP.",
    )

    # SVGP arguments
    parser.add_argument("--fit_svgp", action="store_true", help="Flag to run sparse GP fitting.")
    parser.add_argument(
        "--svgp_num_inducing_points",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--svgp_inducing_init",
        type=str,
        default="kmeans",
        help="Method to initialize SVGP inducing points.",
    )
    # Pretraining (fitting var params only)
    parser.add_argument(
        "--svgp_pretrain_num_steps",
        type=int,
        default=100,
        help="Number of steps to run only variational parameters for.",
    )
    parser.add_argument(
        "--svgp_pretrain_batch_size",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--svgp_pretrain_lr",
        type=float,
        default=1e-1,
    )
    parser.add_argument(
        "--svgp_pretrain_eval_interval",
        type=int,
        default=100,
    )
    # Normal training (inducing points + var params)
    parser.add_argument(
        "--svgp_num_steps",
        type=int,
        default=50_000,
        help="Number of standard optimization steps to run.",
    )
    parser.add_argument(
        "--svgp_batch_size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--svgp_lr",
        type=float,
        default=1e-3,
        help="Learning rate for inducing + var param training.",
    )
    parser.add_argument(
        "--svgp_eval_interval",
        type=int,
        default=1_000,
    )

    # random feature GP arguments
    parser.add_argument("--fit_rfgp", action="store_true", help="Flag to run random feature GP fitting.")
    parser.add_argument(
        "--num_random_features",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--tdp_bias_correction",
        type=str,
        default="normalize",
    )
    parser.add_argument(
        "--tmm_distribution",
        type=str,
        default="Rademacher",
    )
    return parser


def gp_predict_and_eval(
    gp_model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    noise_var: float,
    **kwargs,
) -> dict[str, float]:
    mu_pred, std_pred = batch_predict_mu_std_numpy(gp_model=gp_model, x=X_test, include_std=True, **kwargs)
    return _eval_gp_predictions(mu_pred, std_pred, y_test, noise_var)


def _eval_gp_predictions(mu_pred, std_pred, y_test: np.ndarray, noise_var: float) -> dict[str, float]:
    total_std = np.sqrt(std_pred**2 + noise_var)
    R2 = sklearn.metrics.r2_score(y_true=y_test, y_pred=mu_pred)
    log_probs = stats.norm.logpdf(y_test, loc=mu_pred, scale=total_std)
    mae = sklearn.metrics.mean_absolute_error(y_true=y_test, y_pred=mu_pred)
    mse = sklearn.metrics.mean_squared_error(y_true=y_test, y_pred=mu_pred)
    return dict(R2=R2, avg_log_prob=np.average(log_probs), mae=mae, mse=mse)


def _eval_subset_gp(
    reference_gp,
    rng,
    subset_size: int,
    fp_train: np.ndarray,
    y_train: np.ndarray,
    fp_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, dict[str, float]]:
    t_start = time.monotonic()

    # Create GP
    curr_random_subset_indices = rng.choice(len(fp_train), size=subset_size, replace=False)
    gp = TanimotoKernelGP(
        train_x=torch.as_tensor(fp_train[curr_random_subset_indices]),
        train_y=torch.as_tensor(y_train[curr_random_subset_indices]),
        kernel=reference_gp.covar_module,
        mean_obj=reference_gp.mean_module,
    )

    # Get metrics
    metrics = gp_predict_and_eval(
        gp_model=gp,
        X_test=fp_test,
        y_test=y_test,
        noise_var=gp.likelihood.noise.item(),
        batch_size=GP_EVAL_BATCH_SIZE,
        device="cpu",
    )

    # Time taken
    time_taken = time.monotonic() - t_start

    return time_taken, metrics


def _fit_and_eval_svgp(
    reference_gp,
    rng,
    fp_train: np.ndarray,
    y_train: np.ndarray,
    fp_test: np.ndarray,
    y_test: np.ndarray,
    output: dict,
    args,
) -> None:
    # First step of fitting: initialize inducing points.
    # Good choice of inducing points helps a lot with convergence.
    logger.debug("Initializing inducing points")
    t_start = time.monotonic()
    if args.svgp_inducing_init == "random":
        init_inducing_points_np = fp_train[rng.choice(len(fp_train), size=args.svgp_num_inducing_points, replace=False)]
    elif args.svgp_inducing_init == "kmeans":
        kmeans = KMeans(
            n_clusters=args.svgp_num_inducing_points,
            n_init=1,
            init="random",
            random_state=np.random.RandomState(rng.bit_generator),
        ).fit(fp_train)
        init_inducing_points_np = kmeans.cluster_centers_
        del kmeans
    else:
        raise NotImplementedError()
    time_init_inducing = time.monotonic() - t_start
    output["times"]["init_inducing_points"] = time_init_inducing
    logger.debug(f"Finished initializing inducing points in {time_init_inducing} seconds.")

    # Make SVGP
    svgp = CustomKernelSVGP(
        kernel_obj=reference_gp.covar_module,
        inducing_points=torch.as_tensor(init_inducing_points_np),
        mean_obj=reference_gp.mean_module,
        variational_distribution_class=gpytorch.variational.NaturalVariationalDistribution,
    )

    # Setup optimization
    ngd_pretrain_optimizer = gpytorch.optim.NGD(
        svgp.variational_parameters(),
        num_data=fp_train.shape[0],
        lr=args.svgp_pretrain_lr,
    )
    ngd_main_optimizer = gpytorch.optim.NGD(
        svgp.variational_parameters(),
        num_data=fp_train.shape[0],
        lr=args.svgp_lr,
    )
    inducing_optimizer = torch.optim.Adam(
        svgp.inducing_point_parameters(),
        lr=args.svgp_lr,
    )
    mll = gpytorch.mlls.VariationalELBO(reference_gp.likelihood, svgp, num_data=fp_train.shape[0])

    # Set up datasets for minibatch training (only train on full batches)
    train_dataset = torch.utils.data.TensorDataset(torch.as_tensor(fp_train), torch.as_tensor(y_train))
    torch.manual_seed(args.seed)  # control randomness in dataloader
    dataloader_main = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.svgp_batch_size, shuffle=True, drop_last=True
    )
    dataloader_pretrain = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.svgp_pretrain_batch_size,
        shuffle=True,
        drop_last=True,
    )

    # Training loop
    n_opt_steps = 0
    total_svgp_train_time = 0.0
    output["metrics"]["svgp"] = []
    svgp_train_time_tracker = time.monotonic()
    svgp.train()
    total_opt_steps = args.svgp_num_steps + args.svgp_pretrain_num_steps
    while n_opt_steps < total_opt_steps:
        # Check if pretraining: if so, use pretraining optimizer
        currently_pretraining = n_opt_steps < args.svgp_pretrain_num_steps
        if currently_pretraining:
            logger.debug("Setting up epoch of pretraining.")
            dataloader = dataloader_pretrain
            ngd_optimizer = ngd_pretrain_optimizer
            svgp_eval_interval = args.svgp_pretrain_eval_interval
        else:
            logger.debug("Setting up epoch of full training.")
            dataloader = dataloader_main
            ngd_optimizer = ngd_main_optimizer
            svgp_eval_interval = args.svgp_eval_interval

        # Set requires grad for inducing points
        for p in svgp.inducing_point_parameters():
            p.requires_grad_(not currently_pretraining)

        for x_batch, y_batch in dataloader:
            # Get gradients
            ngd_optimizer.zero_grad()
            inducing_optimizer.zero_grad()
            with gpytorch.settings.fast_computations(False, False, False):
                gp_pred_output = svgp(x_batch)
                loss = -mll(gp_pred_output, y_batch)
            loss.backward()

            # Logging
            if n_opt_steps % 25 == 0:
                logger.debug(f"\tm,S + Z step {n_opt_steps}: loss={loss.item():.3g}")

            # Take optimization step
            ngd_optimizer.step()
            inducing_optimizer.step()
            n_opt_steps += 1

            # Ensure inducing points are non-negative
            with torch.no_grad():
                for p in svgp.inducing_point_parameters():
                    torch.clamp_min_(p, 0.0)

            # Possibly evaluate
            if n_opt_steps % svgp_eval_interval == 0:
                total_svgp_train_time += time.monotonic() - svgp_train_time_tracker
                logger.debug(f"Evaluating SVGP at {n_opt_steps} steps...")
                t_start = time.monotonic()
                svgp.eval()
                svgp_metrics = gp_predict_and_eval(
                    gp_model=svgp,
                    X_test=fp_test,
                    y_test=y_test,
                    noise_var=mll.likelihood.noise.item(),
                    batch_size=GP_EVAL_BATCH_SIZE,
                )
                logger.debug(f"Done evaluating SVGP. Metrics: {pformat(svgp_metrics)}")

                # Save metrics and write output
                output["metrics"]["svgp"].append(
                    {
                        "n_opt_steps": n_opt_steps,
                        "train_time": total_svgp_train_time,
                        "eval_time": time.monotonic() - t_start,
                        "metrics": svgp_metrics,
                    }
                )
                del t_start

                svgp.train()
                svgp_train_time_tracker = time.monotonic()

            # Optionally termnate exactly when pretraining is done
            if currently_pretraining and n_opt_steps == args.svgp_pretrain_num_steps:
                logger.info("Done pretraining SVGP")
                break

            # Terminate exactly when training is done
            if n_opt_steps >= total_opt_steps:
                logger.debug("Breaking out of SVGP training loop")
                break


def _fit_and_eval_rfgp(
    reference_gp,
    rng,
    fp_train: np.ndarray,
    y_train: np.ndarray,
    fp_test: np.ndarray,
    y_test: np.ndarray,
    output: dict,
    args,
) -> None:
    # Compute random features for train and test set
    if args.kernel == "T_DP":
        # Normalize features
        train_norms = np.linalg.norm(fp_train, axis=1)
        normalizer = np.max(train_norms) * 1.2  # bit of padding for safety
        fp_train /= normalizer
        fp_test /= normalizer
        logger.debug(f"Normalizer: {normalizer}")
        logger.debug(
            f"Min norms: train={np.min(np.linalg.norm(fp_train, axis=1)):.3g} "
            f"test={np.min(np.linalg.norm(fp_test, axis=1)):.3g}"
        )

        # Make featurizer
        featurizer = Default_TDP_Featurizer(
            input_dim=fp_train.shape[-1],
            num_rf=args.num_random_features,
            min_input_square_norm=np.min(np.linalg.norm(fp_train, axis=1)) ** 2,
            bias_correction=args.tdp_bias_correction,
            rng=rng,
        )

        def _batch_featurize(arr, batch_size: int = 1000) -> np.ndarray:
            out = []
            with tqdm(
                total=len(arr),
                desc="Featurizing",
            ) as pbar:
                for i in range(0, len(arr), batch_size):
                    out.append(featurizer(arr[i : i + batch_size]))
                    pbar.update(len(out[-1]))
            return np.concatenate(out, axis=0)

        # Train random features
        t_start = time.monotonic()
        random_features_train = _batch_featurize(fp_train)
        output["times"]["make_random_features_train"] = time.monotonic() - t_start

        # Test random features
        t_start = time.monotonic()
        random_features_test = _batch_featurize(fp_test)
        output["times"]["make_random_features_test"] = time.monotonic() - t_start
        del featurizer
    elif args.kernel == "T_MM":
        featurizer = TMM_ArrFeaturizer(  # type: ignore  # confused by Tanimoto featurizer above
            num_features=args.num_random_features,
            max_fp_dim=fp_train.shape[-1],
            rng=rng,
            use_tqdm=True,
            n_jobs=args.num_jobs,
            distribution=args.tmm_distribution,
        )

        # Train random features
        t_start = time.monotonic()
        random_features_train = featurizer(fp_train)
        output["times"]["make_random_features_train"] = time.monotonic() - t_start

        # Test random features
        t_start = time.monotonic()
        random_features_test = featurizer(fp_test)
        output["times"]["make_random_features_test"] = time.monotonic() - t_start

        del featurizer
    else:
        raise NotImplementedError(args.kernel)
    logger.debug(
        "Done making random features. Shapes: " f"train={random_features_train.shape} test={random_features_test.shape}"
    )

    # Make predictions (on CPU)
    logger.debug("Making predictions with RF-GP...")
    random_features_train = torch.as_tensor(random_features_train)
    random_features_test = torch.as_tensor(random_features_test)
    x_query_list = [
        random_features_test[i : i + RF_PREDICT_BATCH_SIZE]
        for i in range(0, len(random_features_test), RF_PREDICT_BATCH_SIZE)
    ]
    gp_mean = reference_gp.mean_module.constant.item()
    t_start = time.monotonic()
    with torch.no_grad():
        mean_covar_tuples = batch_linear_gp_predict(
            x_train=random_features_train,
            y_train=torch.as_tensor(y_train - gp_mean),
            x_query_list=x_query_list,
            kernel_variance=reference_gp.covar_module.outputscale.detach().cpu(),
            noise_variance=reference_gp.likelihood.noise.detach().cpu(),
        )
    mu_pred = (torch.cat([tup[0] for tup in mean_covar_tuples], dim=0).detach().cpu().numpy()) + gp_mean
    var_pred = torch.cat([torch.diag(tup[1]) for tup in mean_covar_tuples], dim=0).detach().cpu().numpy()
    output["times"]["rfgp_predictions"] = time.monotonic() - t_start

    logger.debug("Evaluating RF-GP...")
    output["metrics"]["rfgp"] = _eval_gp_predictions(
        y_test=y_test,
        mu_pred=mu_pred,
        std_pred=np.sqrt(var_pred),
        noise_var=reference_gp.likelihood.noise.item(),
    )
    logger.info(f"Done fitting RF-GP. Metrics: {pformat(output['metrics']['rfgp'])}")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    main()
