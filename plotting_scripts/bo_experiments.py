"""Plots for BO analysis."""

import argparse
import collections
import json
import re
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
    num_data_to_results = collections.defaultdict(list)
    for json_file in Path(args.results_dir).glob("*.json"):
        with open(json_file, "r") as f:
            res = json.load(f)
        num_data_to_results[int(re.match("N(\d+)-.*json", json_file.name).group(1))].append(res)
    num_data_arr = np.array(sorted(num_data_to_results.keys()))

    # Set plot parameters
    plt.rcParams.update(fontsizes.neurips2023())
    plt.rcParams.update(fonts.neurips2023())
    plt.rcParams.update(figsizes.neurips2023(ncols=2, nrows=1, rel_width=1.0, height_to_width_ratio=0.5))

    # Make key plots: dataset size vs time and performance vs time
    method_to_label = {
        "TMM_exact": "$T_{MM}$ (exact)",
        "TMM_rf": "$T_{MM}$ (RFs)",
        "TDP_exact": "$T_{DP}$ (exact)",
        "TDP_rf": "$T_{DP}$ (RFs)",
    }
    fig, axes = plt.subplots(nrows=1, ncols=2)
    for method in method_to_label:
        # Extract results
        times = []
        top_mols = []
        for N in num_data_arr:
            times.append([])
            top_mols.append([])
            for res_list in num_data_to_results[N]:
                for res in res_list:
                    if res["method"] == method:
                        times[-1].append(res["time"])
                        top_mols[-1].append(np.mean(res["y_values"]))
        times = np.asarray(times)
        top_mols = np.asarray(top_mols)
        print(f"{method}: {times.shape} {top_mols.shape}")

        # Times
        plt.sca(axes[0])
        mean_time = np.mean(times, axis=1)
        std_err_time = np.std(times, axis=1) / np.sqrt(times.shape[1])
        plt.loglog(num_data_arr, mean_time, ".-", label=method_to_label[method])
        plt.fill_between(num_data_arr, mean_time - std_err_time, mean_time + std_err_time, alpha=0.3)

        # Scores of molecules
        plt.sca(axes[1])
        mean_score = np.mean(top_mols, axis=1)
        std_err_score = np.std(top_mols, axis=1) / np.sqrt(top_mols.shape[1])
        plt.plot(num_data_arr, mean_score, ".-", label=method_to_label[method])
        plt.fill_between(num_data_arr, mean_score - std_err_score, mean_score + std_err_score, alpha=0.3)
        plt.xscale("log")

    # Labels
    for ax in axes:
        ax.set_xlabel("Number of data points")
    axes[0].set_ylabel("Time (s)")
    axes[1].set_ylabel("Average F2 score")

    fig.legend(*axes[0].get_legend_handles_labels(), bbox_to_anchor=(0.5, -0.0), ncol=5, loc="upper center")
    plt.savefig(output_dir / "bo_results.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
