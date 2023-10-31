"""Make plots for random feature analysis."""

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from tueplots import figsizes, fonts, fontsizes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    assert output_dir.exists()

    # Read all results
    all_results = dict()
    fp_types = ["count", "binary"]
    for kernel in ["tmm", "tdp"]:
        for fp_type in fp_types:
            with open(f"{args.results_dir}/{kernel}_{fp_type}_fp.json", "r") as f:
                all_results[(kernel, fp_type)] = json.load(f)

    # Update fontsizes
    plt.rcParams.update(fontsizes.neurips2023())
    plt.rcParams.update(fonts.neurips2023())

    # TMM plots. Set up a figure with 2 subplots beside each other.
    # Left subplot: MSE w.r.t. number of random features
    # Right subplot: variance w.r.t. T_MM(x,x')
    plt.rcParams.update(
        figsizes.neurips2023(
            ncols=2,
            nrows=1,
            rel_width=1.0,
        )
    )
    fig, axes = plt.subplots(nrows=1, ncols=2)
    axins = axes[0].inset_axes(  # Inset axes for left subplot to zoom in on lines
        [0.7, 0.7, 0.15, 0.2],
        xlim=(9.5e3, 1.05e4),
        ylim=(0.8e-4, 1.6e-4),
    )
    tmm_xi = ["Rademacher", "Gaussian"]
    for fp_type in fp_types:
        for xi in tmm_xi:
            curr_res = all_results[("tmm", fp_type)]
            label = f"{xi}, {fp_type} FPs"

            # MSE
            for ax in [axes[0], axins]:
                kwargs = dict()
                if xi == "Gaussian":
                    kwargs["linestyle"] = "--"
                _plot_mses_vs_num_rf(curr_res, xi, ax, label, **kwargs)
                del kwargs

            # Variance
            plt.sca(axes[1])
            d = curr_res["tmm_variances"][xi]
            plt.plot(d["tmm"], d["var"], ".", label=label)

    # Plot theoretically expected variances on right subplot
    x_plot = np.linspace(0, 1, 100)
    dist_to_expected_y = {
        "Rademacher": 1 - x_plot**2,
        "Gaussian": 1 + 2 * x_plot - x_plot**2,
    }
    plt.sca(axes[1])
    for xi in tmm_xi:
        plt.plot(x_plot, dist_to_expected_y[xi], "k--")

    # Axis labels
    axes[0].indicate_inset_zoom(axins, edgecolor="black")
    _set_mse_rf_axis_labels(axes[0])
    axes[0].set_ylabel("$T_{MM}$ MSE")
    axes[1].set_xlabel("$T_{MM}(x,x')$")
    axes[1].set_ylabel("$T_{MM}$ Variance")
    axins.get_xaxis().set_visible(False)
    axins.get_yaxis().set_visible(False)

    # Place legend for first plot horizontally below both plots
    fig.legend(*axes[0].get_legend_handles_labels(), bbox_to_anchor=(0.5, -0.0), ncol=4, loc="upper center")
    plt.savefig(output_dir / "tmm_variance.pdf")
    plt.close(fig)

    # TDP prefactor plots.
    # Plot 1: MSE vs r,c contour plots
    TDP_R_LIST = [1, 3]
    with plt.rc_context(
        {
            "axes.titlesize": 7,
            **figsizes.neurips2023(ncols=2, nrows=2, rel_width=0.5),
        },
    ):
        fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
        for fp_i, fp_type in enumerate(fp_types):
            curr_res = all_results[("tdp", fp_type)]
            for r_i, r in enumerate(TDP_R_LIST):
                plt.sca(axes[fp_i, r_i])
                arr = np.asarray(curr_res[f"prefactor_r{r}_M10000_sc_mses"])
                y = np.median(arr, axis=0)
                x = curr_res["SC_MULT_ARR"]
                ctr = plt.contourf(np.log10(y), vmin=-9, vmax=2)

                # Add lines for optimal s, c
                middle = len(y) // 2
                plt.axhline(middle, color="white", linestyle="-", linewidth=0.5)
                plt.axvline(middle, color="white", linestyle="-", linewidth=0.5)
                tick_locations = [0, middle, len(x) - 1]
                tick_marks = [f"{x[i]:.1g}" for i in tick_locations]
                plt.xticks(tick_locations, tick_marks)
                plt.yticks(tick_locations, tick_marks)
                if fp_i == 1:
                    plt.xlabel("$s/s^*$")
                if r_i == 0:
                    plt.ylabel("$c/c^*$")
                plt.title(f"r={r} ({fp_type} FP)")
                del arr, x, y, middle
        fig.colorbar(ctr, ax=axes.ravel().tolist(), label=r"Median $\log_{10}{\mathrm{MSE}}$")
        plt.savefig(f"{args.output_dir}/tdp_prefactor_contours.pdf")
        plt.close(fig)

    # Plot of prefactor MSE vs number of random features
    with plt.rc_context(figsizes.neurips2023(ncols=1, nrows=1, rel_width=0.5)):
        fig, axes = plt.subplots(nrows=1, ncols=1)
        for fp_type in fp_types:
            curr_res = all_results[("tdp", fp_type)]
            kwargs = dict()
            if fp_type == "binary":
                kwargs["linestyle"] = "--"
            for r in TDP_R_LIST:
                _plot_mses_vs_num_rf(curr_res, f"prefactor_{r}", axes, f"{fp_type} FP, r={r}", **kwargs)
            del kwargs
        _set_mse_rf_axis_labels(axes)
        axes.set_ylabel("Prefactor MSE")
        plt.legend()
        plt.savefig(output_dir / "prefactor_mse.pdf")
        plt.close(fig)

    # Two part figure: error for polynomial sketch (left) and error for term sketch (right)
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True)
    for fp_type in fp_types:
        curr_res = all_results[("tdp", fp_type)]
        for r in TDP_R_LIST:
            kwargs = dict()
            if fp_type == "binary":
                kwargs["linestyle"] = "--"
            _plot_mses_vs_num_rf(curr_res, key=f"poly_{r}", ax=axes[0], label=f"{fp_type} FP, r={r}", **kwargs)
            _plot_mses_vs_num_rf(curr_res, key=f"tdp-term_{r}", ax=axes[1], label=f"{fp_type} FP, r={r}", **kwargs)
            del kwargs
    fig.legend(*axes[0].get_legend_handles_labels(), bbox_to_anchor=(0.5, -0.0), ncol=4, loc="upper center")
    _set_mse_rf_axis_labels(axes[0])
    axes[1].set_xlabel(NUMBER_OF_RANDOM_FEATURES_STR)
    axes[0].set_ylabel("Polynomial MSE")
    axes[1].set_ylabel("$T_{DP}$ term MSE")
    plt.savefig(output_dir / "poly_and_term_term_mse.pdf")
    plt.close(fig)

    # Two part figure: MSE as function of p (left) and overall MSE with bias correction (right).
    # Make separate figure for each fp type
    for fp_type in fp_types:
        fig, axes = plt.subplots(
            nrows=1,
            ncols=2,
        )
        curr_res = all_results[("tdp", fp_type)]
        p_list = curr_res["P_LIST"]
        R = 4
        for bc in ["none", "normalize", "sketch_error"]:
            _loglog_and_fillbetween(
                ax=axes[0],
                x=p_list,
                arr=np.asarray([curr_res["mses"][f"tdp_{bc}_p{p}_R{R}"]["10000"] for p in p_list]),
                label=dict(none="no bias correction", normalize="normalize", sketch_error="sketch error")[bc],
            )
            _plot_mses_vs_num_rf(curr_res, key=f"tdp_{bc}_p{-1.0}_R{R}", ax=axes[1], label=bc)

        axes[0].set_xscale("linear")
        axes[0].set_yscale("linear")
        _set_mse_rf_axis_labels(axes[1])
        axes[0].set_xlabel("Feature allocation $p$")
        axes[0].set_ylabel("$T_{DP}$ MSE")
        axes[1].set_ylabel("$T_{DP}$ MSE (p=-1)")
        fig.legend(
            *axes[0].get_legend_handles_labels(),
            bbox_to_anchor=(0.5, -0.0),
            ncol=4,
            loc="upper center",
        )
        plt.savefig(output_dir / f"tdp_feat_alloc_and_overall_error_{fp_type}.pdf")
        plt.close(fig)


def _plot_mses_vs_num_rf(curr_res, key, ax, label, **kwargs) -> None:
    """Method to plot MSE results in a consistent way."""
    num_rf = curr_res["num_rf_arr"]
    mses = [curr_res["mses"][key][str(m)] for m in num_rf]
    _loglog_and_fillbetween(ax, num_rf, mses, label, **kwargs)


def _loglog_and_fillbetween(ax, x, arr, label, **kwargs) -> None:
    kwargs.setdefault("linestyle", "-")
    kwargs.setdefault("marker", ".")
    ax.loglog(x, np.median(arr, axis=1), label=label, **kwargs)
    ax.fill_between(x, np.quantile(arr, 0.25, axis=1), np.quantile(arr, 0.75, axis=1), alpha=0.3)


NUMBER_OF_RANDOM_FEATURES_STR = "Number of random features"


def _set_mse_rf_axis_labels(ax) -> None:
    ax.set_xlabel(NUMBER_OF_RANDOM_FEATURES_STR)
    ax.set_ylabel("MSE")


if __name__ == "__main__":
    main()
