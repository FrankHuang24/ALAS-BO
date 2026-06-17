"""
fig1_weierstrass_torture.py

Generate Figure 1 (Real-function fit demo) in the paper.

Task
- Weierstrass(1D): a non-smooth 1D function with cusp-like irregularities.

Procedure
1) Sample N training points x ~ Uniform([0,1]) and evaluate y = f(x) (noise-free).
2) Fit two exact GPs:
   - RBF kernel baseline
   - ALAS (initialized at alpha=1.5)
3) Plot posterior mean curves and report learned alpha + test PLL on a dense grid.

Output
- Saves a PDF to <script_dir>/fig1_weierstrass_seed<seed>.pdf
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
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
from test_functions.weierstrass_torture import WeierstrassTorture
from botorch.models.kernels.stable_kernel import LearnableAlphaStableKernel

plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.8) 
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = ':'
plt.rcParams['grid.alpha'] = 0.6

# train model function
def train_model(train_X, train_Y, kernel_type):
    train_X = train_X.double()
    train_Y = train_Y.double()

    noise_prior = GammaPrior(1.1, 0.05)
    likelihood = GaussianLikelihood(
        noise_prior=noise_prior,
        noise_constraint=GreaterThan(1e-4)
    )

    if kernel_type == 'rbf':
        covar = ScaleKernel(RBFKernel())
    else:
        # 1d alas
        base_kernel = LearnableAlphaStableKernel(num_mixtures=1, num_dims=1)
        with torch.no_grad():
            base_kernel.initialize(raw_alphas=base_kernel.raw_alphas_constraint.inverse_transform(torch.tensor([1.5])))
        covar = ScaleKernel(base_kernel)

    model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        likelihood=likelihood,
        covar_module=covar,
        outcome_transform=Standardize(m=1)
    )
    model.double()

    mll = ExactMarginalLogLikelihood(likelihood, model)
    fit_gpytorch_mll(mll, optimizer_kwargs={'options': {'maxiter': 200}})
    return model

def get_alpha(model):
    try:
        return model.covar_module.base_kernel.alphas.detach().item()
    except:
        return 2.0

# evaluate predictive log likelihood
def eval_pll(model, test_X, test_Y):
    model.eval()
    with torch.no_grad():
        post = model.posterior(test_X, observation_noise=True)
        mu = post.mean.squeeze(-1)
        var = post.variance.squeeze(-1).clamp_min(1e-12)
        pll = torch.distributions.Normal(mu, var.sqrt()).log_prob(test_Y.squeeze(-1))
        return pll.mean().item()

# main
def visualize_comparison(seed=28):
    print(f"--- Running Honest Weierstrass Test (Seed {seed}) ---")
    torch.manual_seed(seed)
    np.random.seed(seed)

    func = WeierstrassTorture(dim=1, seed=seed, cusp_weight=1.5)
    N_SAMPLES = 25
    init_x = torch.rand(N_SAMPLES, 1, dtype=torch.double)
    init_y = func.eval(init_x)
    
    test_x = torch.linspace(0, 1, 500, dtype=torch.double).unsqueeze(-1)
    true_y = func.eval(test_x)

    print("Fitting RBF Baseline...")
    model_rbf = train_model(init_x, init_y, 'rbf')
    print("Fitting ALAS (Ours)...")
    model_alas = train_model(init_x, init_y, 'alas')
    #evaluate pll
    alpha = get_alpha(model_alas)
    pll_rbf = eval_pll(model_rbf, test_x, true_y)
    pll_alas = eval_pll(model_alas, test_x, true_y)
    
    print(f"\nResult: Alpha={alpha:.3f}")
    print(f"Avg Test PLL: RBF={pll_rbf:.3f}, ALAS={pll_alas:.3f}")

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 7), constrained_layout=True)
    
    with torch.no_grad():
        p_rbf = model_rbf.posterior(test_x)
        m_rbf = p_rbf.mean.squeeze(-1)
        
        p_alas = model_alas.posterior(test_x)
        m_alas = p_alas.mean.squeeze(-1)

    # True Function 
    ax.plot(test_x.numpy(), true_y.numpy(), 'k:', linewidth=1.5, alpha=0.5, label='True Function')
    
    # RBF Baseline
    ax.plot(test_x.numpy(), m_rbf.numpy(), color='#d62728', linestyle='--', linewidth=2.5, label='RBF Mean')
    
    # ALAS 
    ax.plot(test_x.numpy(), m_alas.numpy(), color='#1f77b4', linestyle='-', linewidth=3.0, label='ALAS Mean')
    
    # Data Points
    ax.scatter(init_x.numpy(), init_y.numpy(), c='k', marker='x', s=120, linewidth=2, zorder=10, label='Training Data')
    
    # Legend
    ax.legend(loc='lower right', framealpha=0.95, edgecolor='gray', fancybox=True, fontsize=14)

    min_idx = true_y.argmin()
    x_min = test_x[min_idx].item()
    y_min = true_y[min_idx].item()

    ax.scatter([x_min], [y_min], s=300, facecolors='none', edgecolors='red', linewidth=2.5, linestyle='--', zorder=20)

    ax.annotate(
        'Optimal Solution',  
        xy=(x_min, y_min),
        xytext=(x_min - 0.25, y_min),  
        fontsize=14,
        fontweight='bold',
        arrowprops=dict(
            arrowstyle="->", 
            color="black",
            linewidth=1.5,
        )
    )
    
    metrics_text = (
        f"Avg Test PLL:\n"
        f"RBF:  {pll_rbf:.2f}\n"
        f"ALAS: {pll_alas:.2f}\n"
        f"Learned $\\alpha$= {alpha:.2f}"
    )
    ax.text(0.02, 0.05, metrics_text, transform=ax.transAxes, 
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='#cccccc', boxstyle='round,pad=0.5'), 
            fontsize=13, va='bottom', fontfamily='monospace')

    ax.set_xlim(-0.02, 1.02)

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(ax, width="25%", height="17%", loc="upper right", borderpad=1.5)

    diff = m_alas - m_rbf
    axins.plot(test_x.numpy(), diff.numpy(), linewidth=1.5, color="purple")
    axins.axhline(0.0, linestyle=":", color="k", linewidth=1.0)

    axins.set_title(r"Diff ($m_{\mathrm{ALAS}} - m_{\mathrm{RBF}}$)", fontsize=11)
    axins.grid(True, linestyle=":", alpha=0.3)
    axins.tick_params(labelsize=9)

    max_diff = diff.abs().max().item()
    if max_diff < 1e-2:
        axins.set_ylim(-0.01, 0.01) 

    out_path = f"alas_weierstrass_final_seed{seed}.pdf"
    save_path = os.path.join(current_dir, out_path)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    print(f"Plot saved to {save_path}")
if __name__ == "__main__":
    visualize_comparison(seed=28)