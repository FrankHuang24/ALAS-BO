"""
fig2_adaptivity_rbf_truth.py

Generate Figure 2 (Adaptivity demo) in the paper.

Setup
- We sample a 1D "ground-truth" function from a GP prior with an RBF kernel
  (lengthscale_true is configurable). This sampled function is treated as the
  noise-free target function f(x).

Procedure
1) Draw f on a dense grid x_test in [0,1] from the GP prior (RBF truth).
2) Sample N_TRAIN random training inputs and obtain their noise-free outputs
   by interpolating from the dense grid (so training targets are consistent
   with the sampled truth).
3) Fit two exact GPs on the training data:
   - RBF kernel (oracle baseline, matches the truth family)
   - ALAS(initialized at alpha=1.5)
4) Report learned alpha and average test predictive log-likelihood (PLL),
   and plot posterior means against the ground-truth function.

Output
- Saves a PDF figure to the script directory:
    fig2_adaptivity_rbf_seed<seed>.pdf

Usage
  python fig2_adaptivity_rbf_truth.py
"""
import sys
import os
import math
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import GreaterThan
from gpytorch.priors import GammaPrior
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from botorch.models.gp_regression import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms import Standardize
from botorch.models.kernels.stable_kernel import LearnableAlphaStableKernel

plt.style.use("seaborn-v0_8-paper")
sns.set_context("paper", font_scale=1.8)
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.linestyle"] = ":"
plt.rcParams["grid.alpha"] = 0.6

#Kernel Definitions for Ground Truth
def rbf_covariance(x: torch.Tensor, lengthscale: float, outputscale: float) -> torch.Tensor:
    """
    Squared Exponential (RBF) Kernel:
    k(r) = σ² exp( -r² / (2ℓ²) )
    """
    # x: (N, 1) -> dist_sq: (N, N)
    dist_sq = torch.cdist(x, x).pow(2)
    K = outputscale * torch.exp(-dist_sq / (2.0 * lengthscale**2 + 1e-12))
    return K

def matern52_covariance(x: torch.Tensor, lengthscale: float, outputscale: float) -> torch.Tensor:
    """Matérn ν=5/2 kernel"""
    r = torch.cdist(x, x)
    sqrt5 = math.sqrt(5.0)
    t = sqrt5 * r / (lengthscale + 1e-12)
    K = outputscale * (1.0 + t + (t ** 2) / 3.0) * torch.exp(-t)
    return K

def sample_true_function_from_gp(
    test_x: torch.Tensor,
    seed: int,
    kernel: str = "rbf",        
    lengthscale_true: float = 0.10,
    outputscale_true: float = 1.0,
    jitter: float = 1e-6,
) -> torch.Tensor:
    """
    Returns true_y: (N_test, 1)
    """
    torch.manual_seed(seed)

    if kernel == "rbf":
        K = rbf_covariance(test_x, lengthscale_true, outputscale_true)
    elif kernel == "matern52":
        K = matern52_covariance(test_x, lengthscale_true, outputscale_true)
    else:
        raise ValueError(f"Unknown kernel='{kernel}'")
    
    K = 0.5 * (K + K.t()) + jitter * torch.eye(K.shape[0], dtype=K.dtype, device=K.device)

    mvn = torch.distributions.MultivariateNormal(
        loc=torch.zeros(test_x.shape[0], dtype=test_x.dtype, device=test_x.device),
        covariance_matrix=K,
    )
    y = mvn.sample().unsqueeze(-1)
    y = (y - y.mean()) / (y.std() + 1e-12)
    return y


def interp_from_grid(test_x: torch.Tensor, test_y: torch.Tensor, train_x: torch.Tensor) -> torch.Tensor:
    xg = test_x.squeeze(-1)
    yg = test_y.squeeze(-1)

    xt = train_x.squeeze(-1)
    order = torch.argsort(xt)
    xt_sorted = xt[order]

    idx = torch.searchsorted(xg, xt_sorted).clamp(1, xg.numel() - 1)
    x0, x1 = xg[idx - 1], xg[idx]
    y0, y1 = yg[idx - 1], yg[idx]
    w = (xt_sorted - x0) / (x1 - x0 + 1e-12)
    yt_sorted = (1 - w) * y0 + w * y1

    yt = torch.empty_like(xt)
    yt[order] = yt_sorted
    return yt.unsqueeze(-1)

def train_model(train_X, train_Y, kernel_type, maxiter: int = 300):
    train_X = train_X.double()
    train_Y = train_Y.double()

    noise_prior = GammaPrior(1.1, 0.05)
    likelihood = GaussianLikelihood(
        noise_prior=noise_prior,
        noise_constraint=GreaterThan(1e-5),
    )

    if kernel_type == "rbf":
        covar = ScaleKernel(RBFKernel())
        covar.base_kernel.initialize(lengthscale=torch.tensor(0.05, dtype=torch.double))
        covar.initialize(outputscale=torch.tensor(0.5, dtype=torch.double))
    else:
        base_kernel = LearnableAlphaStableKernel(num_mixtures=1, num_dims=1)
        with torch.no_grad():
            #initialize alpha to 1.5, test if it can recover 2.0 for RBF truth
            init_alpha = torch.tensor([1.5], dtype=torch.double)
            base_kernel.initialize(raw_alphas=base_kernel.raw_alphas_constraint.inverse_transform(init_alpha))
        covar = ScaleKernel(base_kernel)

    model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        likelihood=likelihood,
        covar_module=covar,
        outcome_transform=Standardize(m=1),
    ).double()

    mll = ExactMarginalLogLikelihood(likelihood, model)
    fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": maxiter}})

    model.eval()
    model.likelihood.eval()
    return model


def fit_with_restarts(train_X, train_Y, kernel_type, n_restarts=5, maxiter=300, base_seed=0):
    best_state = None
    best_mll_val = -1e30
    best_model = None

    for r in range(n_restarts):
        torch.manual_seed(base_seed + 1000 * r + 999)
        np.random.seed(base_seed + 1000 * r + 999)

        try:
            model = train_model(train_X, train_Y, kernel_type, maxiter=maxiter)
            model.train()
            mll = ExactMarginalLogLikelihood(model.likelihood, model)

            with torch.no_grad():
                out = model(model.train_inputs[0])
                cur = mll(out, model.train_targets).item()

            if cur > best_mll_val:
                best_mll_val = cur
                best_state = copy.deepcopy(model.state_dict())
                best_model = model
        except:
            continue

    if best_model is None:
        return train_model(train_X, train_Y, kernel_type, maxiter=maxiter)

    best_model.load_state_dict(best_state)
    best_model.eval()
    best_model.likelihood.eval()
    return best_model


def get_alpha(model) -> float:
    try:
        bk = model.covar_module.base_kernel
        return float(bk.alphas.detach().cpu().item())
    except Exception:
        return 2.0


def eval_pll(model, test_X, test_Y):
    model.eval()
    with torch.no_grad():
        post = model.posterior(test_X, observation_noise=True)
        mu = post.mean.squeeze(-1)
        var = post.variance.squeeze(-1).clamp_min(1e-12)
        pll = torch.distributions.Normal(mu, var.sqrt()).log_prob(test_Y.squeeze(-1))
        return float(pll.mean().item())


#visualization for Figure 2 (Adaptivity Test with RBF Truth)
def visualize_fig2_adaptivity(seed=12):
    print(f"--- Running Fig2 (Adaptivity: Truth=RBF) | Seed {seed} ---")
    N_TRAIN = 25         
    N_TEST = 800
    TRUE_KERNEL = "rbf"   
    LENGTHSCALE_TRUE = 0.10 
    
    test_x = torch.linspace(0, 1, N_TEST, dtype=torch.double).unsqueeze(-1)
    
    # generate Ground Truth
    true_y = sample_true_function_from_gp(
        test_x,
        seed=seed,
        kernel=TRUE_KERNEL,
        lengthscale_true=LENGTHSCALE_TRUE,
    )

    torch.manual_seed(seed + 100)
    train_x = torch.rand(N_TRAIN, 1, dtype=torch.double)
    train_y = interp_from_grid(test_x, true_y, train_x)

    # Fit Models
    print("Fitting RBF Baseline (The Oracle)...")
    model_rbf = fit_with_restarts(train_x, train_y, "rbf", n_restarts=8, maxiter=300, base_seed=seed)
    
    print("Fitting ALAS (Initialized @ 1.5)...")
    model_alas = fit_with_restarts(train_x, train_y, "alas", n_restarts=8, maxiter=300, base_seed=seed)

    alpha = get_alpha(model_alas)
    pll_rbf = eval_pll(model_rbf, test_x, true_y)
    pll_alas = eval_pll(model_alas, test_x, true_y)

    print(f"\nResult: Learned alpha={alpha:.4f}")
    print(f"Avg Test PLL: RBF={pll_rbf:.4f}, ALAS={pll_alas:.4f}")

    with torch.no_grad():
        m_rbf = model_rbf.posterior(test_x).mean.squeeze(-1)
        m_alas = model_alas.posterior(test_x).mean.squeeze(-1)

    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(12, 7), constrained_layout=True)

    # 1. True Function
    ax.plot(
        test_x.numpy(),
        true_y.numpy(),
        color="0.4",
        linestyle=":",
        linewidth=2.0,
        alpha=0.8,
        label="True Function",
        zorder=1
    )


    #  RBF Mean
    line_rbf, = ax.plot(
        test_x.numpy(),
        m_rbf.numpy(),
        color="#d62728",
        linestyle="--",
        linewidth=2.5,
        label="RBF Mean",
        zorder=4,
    )
    #  1D ALAS Mean
    ax.plot(
        test_x.numpy(),
        m_alas.numpy(),
        color="#1f77b4",
        linestyle="-",
        linewidth=3.0,
        label="ALAS Mean",
        zorder=3,
        alpha=0.9
    )
    line_rbf.set_path_effects([pe.Stroke(linewidth=4.5, foreground="white"), pe.Normal()])

    # 4. Training Data
    ax.scatter(
        train_x.numpy(),
        train_y.numpy(),
        c="k",
        marker="x",
        s=120,
        linewidth=2,
        zorder=10,
        label="Training Data",
    )

    min_idx = true_y.argmin()
    min_x_val = test_x[min_idx].item()
    min_y_val = true_y[min_idx].item()

    ax.scatter([min_x_val], [min_y_val], s=300, facecolors='none', edgecolors='red', linewidth=2.5, linestyle='--', zorder=20)

    ax.annotate(
        'Optimal Solution',  
        xy=(min_x_val, min_y_val), 
        xytext=(min_x_val + 0.1, min_y_val),  
        fontsize=14,
        fontweight='bold',
        arrowprops=dict(
            arrowstyle="->",
            color="black",
            linewidth=1.5, 
        )
    )
    from matplotlib.ticker import MultipleLocator
    ax.yaxis.set_major_locator(MultipleLocator(1))


    # Inset: Difference Plot 
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(ax, width="25%", height="17%", loc="upper right", borderpad=1.5)
    diff = m_alas - m_rbf
    axins.plot(test_x.numpy(), diff.numpy(), linewidth=1.5, color="purple")
    axins.axhline(0.0, linestyle=":", color="k", linewidth=1.0)
    axins.set_title(r"Diff ($m_{\mathrm{ALAS}} - m_{\mathrm{RBF}}$)", fontsize=11)
    axins.grid(True, linestyle=":", alpha=0.3)
    axins.tick_params(labelsize=9)
    max_diff = diff.abs().max().item()
    axins.set_ylim(-0.2, 0.2)

    # Metrics Box
    metrics_text = (
        f"Avg Test PLL:\n"
        f"RBF:  {pll_rbf:.3f}\n"
        f"ALAS: {pll_alas:.3f}\n"
        f"learned ${{\\alpha}} = {alpha:.3f}$"
    )
    ax.text(
        0.02, 0.05, metrics_text,
        transform=ax.transAxes,
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="#cccccc", boxstyle="round,pad=0.5"),
        fontsize=13,
        va="bottom",
        fontfamily="monospace",
    )

    ax.set_xlim(-0.02, 1.02)
    ax.legend(loc="lower right", framealpha=0.95, edgecolor="gray", fancybox=True, fontsize=14)

    #plt.tight_layout()
    out_path = f"fig2_adaptivity_rbf_seed{seed}.pdf"
    save_path = os.path.join(current_dir, out_path)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    visualize_fig2_adaptivity(seed=12)
