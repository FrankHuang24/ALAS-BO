"""
bench_weierstrass_pll_rmse.py

Kernel-fit benchmark script used to generate the summary statistics / LaTeX table
for the 1D WeierstrassTorture task.

What this script does
- For each random seed, sample n_samples training points x ~ Uniform([0,1]) and obtain
  noise-free targets y = f(x) from Weierstrass(1D).
- Fit an exact GP (SingleTaskGP + GaussianLikelihood) with one of the kernels:
  RBF, Matérn-{1/2, 3/2, 5/2}, and ALAS (LearnableAlphaStableKernel with num_mixtures=1).
- Evaluate predictive quality on a dense test grid using:
  (i) RMSE between posterior mean and the noise-free ground truth;
  (ii) average predictive log-likelihood (PLL) under Normal(mu(x), var(x)),
       computed on the latent function posterior (observation_noise=False).

Outputs
- Prints a table (mean ± std), plus a paired comparison for ALAS:
  win rate and ΔPLL vs the per-seed best baseline among the Matérn/RBF kernels.
"""
import os
import sys
import warnings
import numpy as np
import torch

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import GreaterThan
from gpytorch.priors import GammaPrior
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.models.gp_regression import SingleTaskGP 
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms import Standardize
from test_functions.weierstrass_torture import WeierstrassTorture
from botorch.models.kernels.stable_kernel import LearnableAlphaStableKernel


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def eval_metrics(model, test_X, test_Y, var_floor=1e-12):
    """
    Evaluate predictive log-likelihood (PLL) and RMSE against noise-free ground truth.
    Note: we evaluate latent f (observation_noise=False) since test_Y is noise-free.
    """
    model.eval()
    with torch.no_grad():
        post = model.posterior(test_X, observation_noise=False)
        mu = post.mean.squeeze(-1)
        var = post.variance.squeeze(-1).clamp_min(var_floor)

        # PLL under Normal(mu, var)
        pll = torch.distributions.Normal(mu, var.sqrt()).log_prob(test_Y.squeeze(-1))
        avg_pll = pll.mean().item()

        mse = (mu - test_Y.squeeze(-1)).pow(2).mean().item()
        rmse = float(np.sqrt(mse))

    return float(avg_pll), rmse


def get_learned_param(model, kernel_type):
    """
    For ALAS: alpha
    For RBF/Matern: lengthscale (mean if ARD vector)
    """
    try:
        if kernel_type == "alas":
            # ScaleKernel(base_kernel) -> base_kernel.alphas
            a = model.covar_module.base_kernel.alphas.detach().cpu().view(-1).mean().item()
            return float(a)
        else:
            # ScaleKernel(RBF/Matern) -> base_kernel.lengthscale
            ls = model.covar_module.base_kernel.lengthscale.detach().cpu().view(-1).mean().item()
            return float(ls)
    except Exception:
        return np.nan


def train_model(train_X, train_Y, kernel_type, maxiter=200, init_alpha=1.5):
    """
    Train SingleTaskGP with specified kernel.
    Uses standard fit_gpytorch_mll with L-BFGS-B (via SciPy) through BoTorch.
    """
    train_X = train_X.double()
    train_Y = train_Y.double()

    # Likelihood (match your previous setting)
    noise_prior = GammaPrior(1.1, 0.05)
    likelihood = GaussianLikelihood(
        noise_prior=noise_prior,
        noise_constraint=GreaterThan(1e-4),
    )

    d = train_X.shape[-1]

    if kernel_type == "rbf":
        base = RBFKernel(ard_num_dims=d)
        covar = ScaleKernel(base)

    elif kernel_type == "matern32":
        base = MaternKernel(nu=1.5, ard_num_dims=d)
        covar = ScaleKernel(base)

    elif kernel_type == "matern52":
        base = MaternKernel(nu=2.5, ard_num_dims=d)
        covar = ScaleKernel(base)
    
    elif kernel_type == "matern12":
        base = MaternKernel(nu=0.5, ard_num_dims=d)
        covar = ScaleKernel(base)

    elif kernel_type == "alas":
        base = LearnableAlphaStableKernel(num_mixtures=1, num_dims=d)
        # Initialize alpha (optional but helpful)
        with torch.no_grad():
            base.initialize(
                raw_alphas=base.raw_alphas_constraint.inverse_transform(
                    torch.tensor([init_alpha], dtype=torch.double)
                )
            )
        covar = ScaleKernel(base)

    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")

    model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        likelihood=likelihood,
        covar_module=covar,
        outcome_transform=Standardize(m=1),
    ).double()

    mll = ExactMarginalLogLikelihood(likelihood, model)
    fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": maxiter, "disp": False}})
    return model


# 3) Main benchmark
def run_benchmark(
    num_seeds=10,
    n_samples=25,
    n_test=500,
    var_floor=1e-12,
    pll_floor=-20.0,     
    use_clip_for_summary=True,
):
    """
    Runs paired evaluation across seeds on WeierstrassTorture(1D).
    Models: RBF, Matern-3/2, Matern-5/2, ALAS.
    """
    torch.set_default_dtype(torch.float64)

    models = ["rbf", "matern32", "matern52", "alas","matern12"]
    history = {m: {"pll": [], "rmse": [], "param": []} for m in models}

    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Data generation
        func = WeierstrassTorture(dim=1, seed=seed, cusp_weight=1.5)
        init_x = torch.rand(n_samples, 1, dtype=torch.double)
        init_y = func.eval(init_x)

        test_x = torch.linspace(0, 1, n_test, dtype=torch.double).unsqueeze(-1)
        true_y = func.eval(test_x)

        for m in models:
            try:
                model = train_model(init_x, init_y, m, maxiter=200, init_alpha=1.5)
                pll, rmse = eval_metrics(model, test_x, true_y, var_floor=var_floor)
                param = get_learned_param(model, m)
            except Exception:
                pll, rmse, param = np.nan, np.nan, np.nan

            history[m]["pll"].append(pll)
            history[m]["rmse"].append(rmse)
            history[m]["param"].append(param)

    # Convert to arrays
    pll = {m: np.array(history[m]["pll"], dtype=float) for m in models}
    rmse = {m: np.array(history[m]["rmse"], dtype=float) for m in models}
    param = {m: np.array(history[m]["param"], dtype=float) for m in models}

    # -------- Summary stats (optionally robust via clipping)
    def mean_std(x):
        x = x[~np.isnan(x)]
        if len(x) == 0:
            return (np.nan, np.nan)
        if len(x) == 1:
            return (float(x[0]), 0.0)
        return (float(np.mean(x)), float(np.std(x, ddof=1)))

    pll_for_stats = {}
    for m in models:
        x = pll[m].copy()
        if use_clip_for_summary:
            x = np.clip(x, pll_floor, None)
        pll_for_stats[m] = x

    stats = {}
    for m in models:
        stats[m] = {
            "pll": mean_std(pll_for_stats[m]),
            "rmse": mean_std(rmse[m]),
            "param": mean_std(param[m]),
        }

    # -------- Paired comparison: ALAS vs best baseline (per-seed best of {RBF, M32, M52})
    baseline_set = ["rbf", "matern32", "matern52", "matern12"]

    # best baseline PLL for each seed (use same "stats PLL" version if you want robust pairing)
    alas_pll_pair = pll_for_stats["alas"]
    baseline_best_pll = np.nanmax(np.vstack([pll_for_stats[b] for b in baseline_set]), axis=0)

    paired_mask = ~np.isnan(alas_pll_pair) & ~np.isnan(baseline_best_pll)
    delta = alas_pll_pair[paired_mask] - baseline_best_pll[paired_mask]

    n_pair = int(np.sum(paired_mask))
    win_rate = float(100.0 * np.mean(delta > 0.0)) if n_pair > 0 else np.nan
    delta_med = float(np.median(delta)) if n_pair > 0 else np.nan
    delta_p10 = float(np.quantile(delta, 0.10)) if n_pair > 0 else np.nan
    delta_p90 = float(np.quantile(delta, 0.90)) if n_pair > 0 else np.nan
    print("\n=== Results Summary (Robust stats in code; table will be plain mean±std) ===")
    print(f"Paired seeds: {n_pair}/{num_seeds}")
    print(f"ALAS vs best baseline win rate (PLL): {win_rate:.1f}%")
    print(f"ΔPLL median [10,90]%: {delta_med:.3f} [{delta_p10:.3f}, {delta_p90:.3f}]")
    for m in models:
        mu_rmse, sd_rmse = stats[m]["rmse"]
        mu_pll, sd_pll = stats[m]["pll"]
        mu_p, sd_p = stats[m]["param"]
        p_name = "alpha" if m == "alas" else "ls"
        print(f"{m:8s}  RMSE {mu_rmse:.3f}±{sd_rmse:.3f} | PLL {mu_pll:.3f}±{sd_pll:.3f} | {p_name} {mu_p:.3f}±{sd_p:.3f}")

    print(r"\begin{table}[t]")
    print(r"\caption{Paired comparison on 1D Weierstrass over $N=10$ seeds. We report test RMSE and average predictive log-likelihood (PLL) against noise-free ground truth (mean $\pm$ std over seeds). For ALAS, we additionally report the paired win rate and the paired $\Delta$PLL relative to the \emph{best} baseline among RBF/Matérn-3/2/Matérn-5/2 for each seed.}")
    print(r"\label{tab:weierstrass_stats}")
    print(r"\begin{center}")
    print(r"\begin{small}")
    print(r"\begin{sc}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"Model & Test RMSE $\downarrow$ & Test PLL $\uparrow$ & Win Rate & $\Delta$PLL (Med [10,90]\%) \\")
    print(r"\midrule")

    def row(model_key, display_name):
        rm_mu, rm_sd = stats[model_key]["rmse"]
        pl_mu, pl_sd = stats[model_key]["pll"]
        if model_key == "alas":
            wr = f"{win_rate:.0f}\\%"
            dpll = f"{delta_med:+.2f} [{delta_p10:+.2f}, {delta_p90:+.2f}]"
            rm = f"\\textbf{{{rm_mu:.3f} $\\pm$ {rm_sd:.3f}}}"
            pl = f"\\textbf{{{pl_mu:.2f} $\\pm$ {pl_sd:.2f}}}"
        else:
            wr = r"--"
            dpll = r"--"
            rm = f"{rm_mu:.3f} $\\pm$ {rm_sd:.3f}"
            pl = f"{pl_mu:.2f} $\\pm$ {pl_sd:.2f}"
        print(f"{display_name} & {rm} & {pl} & {wr} & {dpll} \\\\")

    row("rbf", "RBF")
    row("matern32", r"Mat\'ern-3/2")
    row("matern52", r"Mat\'ern-5/2")
    row("matern12", r"Mat\'ern-1/2")
    row("alas", "ALAS (Ours)")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{sc}")
    print(r"\end{small}")
    print(r"\end{center}")
    print(r"\vskip -0.1in")
    print(r"\end{table}")


if __name__ == "__main__":
    run_benchmark(
        num_seeds=10,
        n_samples=25,
        n_test=500,
        var_floor=1e-12,
        pll_floor=-20.0,
        use_clip_for_summary=True,  
    )
