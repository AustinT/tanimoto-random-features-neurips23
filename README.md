# Tanimoto Random Features

[![Paper](http://img.shields.io/badge/paper-arxiv.2306.14809-B31B1B.svg)](https://arxiv.org/abs/2306.14809)
[![Conference](http://img.shields.io/badge/NeurIPS-2023-4b44ce.svg)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6a69d44b3386e50c06f7107ef4f29302-Abstract-Conference.html)
![GitHub](https://img.shields.io/github/license/AustinT/tanimoto-random-features-neurips23)

This repository provides code to accompany the paper
_Tanimoto Random Features for Scalable Molecular Machine Learning_
published at NeurIPS 2023.
It contains a minimal python implementation of the methods
described in the paper, code to reproduce the experimental results,
all numerical results presented in the paper,
and code to reproduce the plots.

The purpose of this code is to reproduce the results of the paper in a simple way,
_not_ to provide the best possible Tanimoto random features. This means:

1. The code will not be updated with future improvements to the method
  (e.g. newer, more accurate random features),
  since such improvements are not part of the original paper.
2. If you wish to deploy Tanimoto random features in practice, you should probably
  modify/improve this code (even though it could be a good starting point).

If you use this code or wish to deploy it, feel free to contact Austin via email or open a GitHub issue.

## Code overview

The layout of this repository is as follows:

- `trf23/`: a minimal python package implementing Tanimoto random features and Tanimoto Gaussian processes
- `tests/`: code to test `trf23`
- `experiment_scripts/`: main python scripts to run experiments
- `official_results/`: the results of all experiments performed in this paper
- `plotting_scripts/`: python scripts to make plots

## Running instructions

### Experiments

First, set up a python environment.
We provide two files to help with this:

1. `environment.yml`: a minimal conda environment file
2. `environment-exact.yml`: the exact conda environment used to run the experiments

The easiest thing to do is create a new `conda` environment:

```bash
conda env create -f environment.yml
conda activate trf23
```

However, feel free to set up the environment any way you like:
the results should not be highly dependent on which exact versions of the packages are used.
The remaining instructions assume you have a python environment set up.

To check your python environment, you can run [tests](#testing).

#### Random feature analysis

To reproduce the random feature analysis, run:

```bash
bash run_random_feature_analysis.sh
```

This will write outputs to `results/random_feature_analysis`

#### Regression

The arguments for the regression experiments are more complicated,
so we provide a python script which _prints_ the commands to launch the experiments.
The experiments can be run in parallel, for example by running:

```bash
python print_regression_expt_commands.py | xargs -I {} -P 2 bash -c {}
```

This will write outputs to `results/regression`

#### Bayesian optimization

Similar to the regression experiments,
these experiments can be launched by running a script to print commands:

```bash
bash print_bo_expt_commands.py |  xargs -I {} -P 2 bash -c {}
```

### Plotting

Plots can be generated using the following commands:

```bash
python plotting_scripts/random_feature_analysis.py --results_dir official_results/random_feature_analysis --output_dir plots/  # random features
python plotting_scripts/tabulate_regression_results.py --results_dir official_results/regression --output_dir plots/  # regression
python plotting_scripts/bo_experiments.py --results_dir official_results/bo/F2 --output_dir plots/  # BO
```

## Citation

If you find this work useful we would appreciate a citation!
Until the NeurIPS 2023 proceedings are released, feel free to cite our arXiv version:

```bibtex
@article{tripp2023tanimoto,
  title={Tanimoto Random Features for Scalable Molecular Machine Learning},
  author={Tripp, Austin and Bacallado, Sergio and Singh, Sukriti and Hern{\'a}ndez-Lobato, Jos{\'e} Miguel},
  journal={arXiv preprint arXiv:2306.14809},
  year={2023}
}
```

## Development

Although this repo is unlikely to be actively developed,
we nonetheless encourage the use of pre-commit and testing.

### Formatting

Use pre-commit to enforce formatting, large file checks, etc.

If not already installed in your environment, run:

```bash
conda install pre-commit
```

To install the precommit hooks:

```bash
pre-commit install
```

### Testing

We use `pytest` to run tests.
Install pytest and run:

```bash
python -m pytest tests/
```
