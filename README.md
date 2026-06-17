# ALAS: Additive Learnable Alpha-Stable Kernels for Flexible Bayesian Optimization

Code for the ICML 2026 paper **ALAS: Additive Learnable Alpha-Stable Kernels for Flexible Bayesian Optimization**.

ALAS is a stationary exact-GP kernel family for Bayesian optimization. It learns an alpha-stable spectral tail parameter, which controls the effective roughness of the covariance, and a modulation frequency, which captures oscillatory or non-monotone correlations. ALAS-Sep extends this idea with additive one-dimensional components so different input dimensions can learn different tail behavior.

## Repository Layout

| Path | Description |
| --- | --- |
| `run_stable.py` | Main entry point for BO experiments. |
| `botorch/models/kernels/stable_kernel.py` | ALAS and ALAS-Sep kernel implementations. |
| `botorch/models/gp_regression.py` | Modified `SingleTaskGP` kernel dispatch used by the experiments. |
| `utils/utils.py` | One-step BO loop: GP fitting, acquisition construction, and acquisition optimization. |
| `test_functions/` | Synthetic benchmarks and optional application benchmarks. |
| `plot_res/` | Plotting and analysis scripts for saved experiment outputs. |
| `exp_res_best/pkl/` | Generated result files. This directory is created by runs and is ignored by git. |

The repository vendors the BoTorch source tree used in the experiments. The main ALAS-specific changes are concentrated in `stable_kernel.py`, `gp_regression.py`, `utils/utils.py`, and `run_stable.py`.

## Installation

Create a fresh environment and install the core dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Run commands from the repository root so the vendored BoTorch modules are imported:

```bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

Optional dependencies for figure regeneration, portfolio, robot, and HPO tasks are listed as commented entries in `requirements.txt`.

## Quick Start

Run ALAS on Branin with expected improvement:

```bash
python run_stable.py \
  --function_name branin2 \
  --kernel learnable_alpha_stable \
  --n_mixture 1 \
  --acq ei \
  --n_init 5 \
  --n_iter 20 \
  --seed 3
```

Run ALAS-Sep on Hartmann-3:

```bash
python run_stable.py \
  --function_name hartmann3 \
  --kernel additive_stable \
  --n_mixture 1 \
  --acq ei \
  --n_init 10 \
  --n_iter 50 \
  --seed 5
```

In `run_stable.py`, `--seed N` runs seeds `0, ..., N-1`. All benchmark runs use minimization, and the saved trajectory records the best-so-far objective value.

## Kernels

Common kernel choices for `--kernel` include:

| Argument | Meaning |
| --- | --- |
| `learnable_alpha_stable` | ALAS, the coupled learnable alpha-stable kernel. |
| `additive_stable` | ALAS-Sep, the additive per-dimension ALAS kernel. |
| `rbf` | Squared-exponential kernel. |
| `matern12`, `matern32`, `matern52`, `ma52` | Matern baselines. |
| `rq` | Rational quadratic kernel. |
| `gsm` | Gaussian spectral mixture baseline. |
| `csm` | Cauchy spectral mixture baseline. |
| `sdk` | Spectral delta kernel baseline. |

The GP wrapper also accepts the shorter aliases `alas` and `alas_sep`. The examples above use the longer names to match the original experiment filenames.

Supported acquisition functions are `ei`, `pi`, and `ucb`.

## Outputs

Results are written to `exp_res_best/pkl/`. For ALAS with `Q` mixtures, the main output filename has the form

```text
<function_name>_learnable_alpha_q<Q>_<acq>.pkl
```

For ALAS-Sep, the filename has the form

```text
<function_name>_additive_stable_q<Q>_<acq>.pkl
```

When alpha statistics are logged, an additional file is saved with the suffix `_alpha_history.pkl`.

## Plotting and Analysis

Plotting scripts read saved `.pkl` files from `exp_res_best/pkl/`.
Install the optional figure dependencies in `requirements.txt` before running these scripts.

```bash
python plot_res/plot_main_grid.py
python plot_res/analyze_alphas.py
python plot_res/bench_weierstrass_stats.py
```

One-dimensional diagnostic figures can be regenerated with:

```bash
python plot_res/fig1_weierstrass_torture.py
python plot_res/fig2_adaptivity_rbf_truth.py
```

## Optional Benchmarks

The synthetic benchmarks are the recommended starting point for checking the code. Portfolio, robot, and HPO benchmarks may require additional dependencies or data files and can be excluded from a minimal release if licensing or size constraints apply.

## Release Notes

This repository is intended to be a code release rather than the full paper workspace. Generated experiment files, rendered figures, manuscript PDFs, presentation files, and submission agreements should not be committed. The `.gitignore` keeps these local artifacts out of version control while retaining the source scripts needed to reproduce experiments and figures.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{huang2026alas,
  title = {ALAS: Additive Learnable Alpha-Stable Kernels for Flexible Bayesian Optimization},
  author = {Huang, Weibo and Hua, Cheng},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year = {2026},
  note = {Accepted}
}
```
