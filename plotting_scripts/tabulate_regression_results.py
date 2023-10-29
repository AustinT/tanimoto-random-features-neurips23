"""Script to read in and tabulate all regression results."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DOCKSTRING_TARGETS = ["ESR2", "F2", "KIT", "PARP1", "PGR"]
M = 5000
K_LIST = ["T_MM", "T_DP"]
K_TO_RFGP_OPTIONS = {
    "T_MM": ["Rademacher", "Gaussian"],
    "T_DP": ["none", "normalize", "sketch_error"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    assert output_dir.exists()

    # Read all results
    method_to_r2 = defaultdict(list)
    method_to_logp = defaultdict(list)
    for target in DOCKSTRING_TARGETS:
        res_dir = Path(args.results_dir) / f"M{M}" / "countTrue" / target

        for k in K_LIST:
            # SVGP and random subset GP
            for res in res_dir.glob(f"svgp-{k}*.json"):
                with open(res, "r") as f:
                    data = json.load(f)

                # Random subset
                d = data["metrics"]["eval_subset_gp"][f"0;{M}"]
                method_to_r2[(target, k, "RSGP")].append(d["R2"])
                method_to_logp[(target, k, "RSGP")].append(d["avg_log_prob"])

                # SVGP
                d = data["metrics"]["svgp"][0]["metrics"]
                method_to_r2[(target, k, "SVGP")].append(d["R2"])
                method_to_logp[(target, k, "SVGP")].append(d["avg_log_prob"])

            # RFGP
            for xi in K_TO_RFGP_OPTIONS[k]:
                for res in res_dir.glob(f"rfgp-{xi}-{k}*.json"):
                    with open(res, "r") as f:
                        data = json.load(f)

                    d = data["metrics"]["rfgp"]
                    method_to_r2[(target, k, f"RFGP-{xi}")].append(d["R2"])
                    method_to_logp[(target, k, f"RFGP-{xi}")].append(d["avg_log_prob"])

    # Write to file
    for metric, res_dict in [("R2", method_to_r2), ("logp", method_to_logp)]:
        lines = [
            r"\begin{tabular}",
            "{ll" + "r@{\hspace{0.02cm}$\pm$\hspace{0.02cm}}l@{\hspace{0.30cm}}" * len(DOCKSTRING_TARGETS) + "}",
            r"\toprule",
            "Kernel & Method & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{t}}}" for t in DOCKSTRING_TARGETS) + r"\\",
        ]

        # Results for each kernel
        for kern in K_LIST:
            lines.append(r"\midrule")
            keys = [k[-1] for k in res_dict.keys() if k[0] == DOCKSTRING_TARGETS[0] and k[1] == kern]
            for i_k, k in enumerate(keys):
                # Optionally append kernel name
                if i_k == 0:
                    tokens = [dict(T_MM="$T_{MM}$", T_DP="$T_{DP}$")[kern]]
                else:
                    tokens = [""]

                # Append method name, with some renaming
                if k == "RSGP":
                    tokens.append("Rand subset GP")
                elif k == "SVGP":
                    tokens.append(k)
                else:
                    assert k.startswith("RFGP")
                    tokens.append(
                        "RFGP ("
                        + {
                            "Rademacher": "$\\Xi$ Rad.",
                            "Gaussian": "$\\Xi$ Gauss.",
                            "none": "plain",
                            "normalize": "norm",
                            "sketch_error": "sketch",
                        }[k.split("-")[1]]
                        + ")"
                    )

                for target in DOCKSTRING_TARGETS:
                    tokens.append(f"{np.mean(res_dict[(target, kern, k)]):.3f}")
                    tokens.append(f"{np.std(res_dict[(target, kern, k)]):.3f}")
                lines.append(" & ".join(tokens) + r"\\")

        lines += [
            r"\bottomrule",
            r"\end{tabular}",
        ]

        with open(output_dir / f"tabulate_regression_results_{metric}.tex", "w") as f:
            f.write("\n".join(lines))


if __name__ == "__main__":
    main()
