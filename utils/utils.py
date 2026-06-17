"""
Experiment utilities for Bayesian Optimization.

This module provides a single entry point `get_next_points` that fits a SingleTaskGP
(with either standard kernels or our ALAS / ALAS-Sep kernels), constructs an acquisition
function (EI/PI/UCB; minimization by default), and optimizes it to propose next candidate(s).
Optional Plotly helpers are included for 1D inspection/visualization and are not required
for running the main experiments.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.module import Module
from gpytorch.priors import GammaPrior

from botorch.acquisition import (
    ExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.optim import optimize_acqf


# Optional plotting utilities for a lightweight public release.
try:
    import plotly.graph_objects as go  # type: ignore
except Exception:
    go = None


def _unwrap_base_kernel(k: Module) -> Module:
    """Unwrap ScaleKernel / other wrappers to reach the base kernel."""
    while hasattr(k, "base_kernel"):
        k = k.base_kernel  # type: ignore[attr-defined]
    return k


def _weighted_alpha(alphas: np.ndarray, weights: Optional[np.ndarray]) -> Optional[float]:
    if alphas.size == 0:
        return None
    if weights is None or weights.size == 0:
        return float(np.mean(alphas))
    q = min(len(alphas), len(weights))
    a = alphas[:q]
    w = weights[:q]
    s = float(np.sum(w))
    if s <= 1e-12:
        return float(np.mean(a))
    return float(np.sum(w * a) / s)


def extract_alpha_snapshot(model: SingleTaskGP) -> Dict[str, Any]:
    """Extract a lightweight, JSON-serializable snapshot of learned hyperparameters."""
    cov = model.covar_module

    outputscale = 1.0
    if hasattr(cov, "outputscale"):
        try:
            outputscale = float(cov.outputscale.item())
        except Exception:
            outputscale = 1.0

    noise = 0.0
    try:
        noise = float(model.likelihood.noise.item())
    except Exception:
        noise = float("nan")

    base = _unwrap_base_kernel(cov)

    snap: Dict[str, Any] = {
        "kernel_base": type(base).__name__,
        "outputscale": float(outputscale),
        "noise": float(noise),
        "alpha_bar": None,
    }

    # Additive (ALAS-Sep): per-dim kernels stored in base.kernels
    if hasattr(base, "kernels"):
        alpha_per_dim: List[float] = []
        for kd in base.kernels:  # type: ignore[attr-defined]
            kd0 = _unwrap_base_kernel(kd)
            if not hasattr(kd0, "alphas"):
                continue
            try:
                a = kd0.alphas.detach().cpu().numpy().reshape(-1)  # type: ignore[attr-defined]
            except Exception:
                continue
            w = None
            if hasattr(kd0, "mixture_weights"):
                try:
                    w = kd0.mixture_weights.detach().cpu().numpy().reshape(-1)  # type: ignore[attr-defined]
                except Exception:
                    w = None
            abar = _weighted_alpha(a, w)
            if abar is not None:
                alpha_per_dim.append(float(abar))

        snap["kernel_type"] = "alas_sep"
        snap["alpha_per_dim"] = [float(x) for x in alpha_per_dim]
        snap["alpha_bar"] = float(np.mean(alpha_per_dim)) if len(alpha_per_dim) > 0 else None
        return snap

    # Coupled ALAS: weighted alpha over mixtures
    if hasattr(base, "alphas") and hasattr(base, "mixture_weights"):
        try:
            a = base.alphas.detach().cpu().numpy().reshape(-1)  # type: ignore[attr-defined]
            w = base.mixture_weights.detach().cpu().numpy().reshape(-1)  # type: ignore[attr-defined]
            abar = _weighted_alpha(a, w)
            snap["kernel_type"] = "alas"
            snap["alphas"] = [float(x) for x in a.tolist()]
            snap["weights"] = [float(x) for x in w.tolist()]
            snap["alpha_bar"] = float(abar) if abar is not None else None
            return snap
        except Exception:
            pass

    snap["kernel_type"] = "other"
    return snap


def robust_fit_mll(
    mll: ExactMarginalLogLikelihood,
    model: SingleTaskGP,
    *,
    maxiter_lbfgs: int = 50,
    adam_steps: int = 100,
    adam_lr: float = 0.01,
    verbose: bool = True,
) -> None:
    """Best-effort model fit: try BoTorch default (SciPy L-BFGS-B), fallback to Adam."""
    model.train()

    def _safe_loss() -> float:
        try:
            out = model(model.train_inputs[0])
            return float((-mll(out, model.train_targets)).item())
        except Exception:
            return float("nan")

    loss_before = _safe_loss()
    if verbose:
        print(f"  [Model Fit] Start Loss: {loss_before:.5f}")

    success = False
    try:
        fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": int(maxiter_lbfgs), "disp": False}})
        success = True
    except Exception as e:
        if verbose:
            print(f"  [Model Fit] L-BFGS-B failed: {type(e).__name__}")

    loss_now = _safe_loss()
    need_adam = (not success) or (not np.isfinite(loss_now)) or (loss_now > 1e6)

    if need_adam:
        if verbose:
            print("  [Model Fit] Switching to fallback Adam optimization...")
        opt = torch.optim.Adam([{"params": model.parameters()}], lr=float(adam_lr))

        for i in range(int(adam_steps)):
            opt.zero_grad(set_to_none=True)
            try:
                out = model(model.train_inputs[0])
                loss = -mll(out, model.train_targets)
                if torch.isnan(loss) or torch.isinf(loss):
                    if verbose:
                        print(f"    Adam hit NaN/Inf at iter {i}. Stopping.")
                    break
                loss.backward()
                opt.step()
            except Exception as e:
                if verbose:
                    print(f"    Adam error at iter {i}: {type(e).__name__}")
                break

    model.eval()
    if verbose:
        loss_final = _safe_loss()
        print(f"  [Model Fit] End Loss: {loss_final:.5f}")


def get_next_points(
    acq: str,
    kernel: Union[str, Module],
    n_mixture: Optional[int],
    init_x: torch.Tensor,
    init_y: torch.Tensor,
    best_init_y: float,
    bounds: torch.Tensor,
    n_points: int = 1,
    n_mixture1: int = 1,
    n_mixture2: int = 1,
    alphas: Optional[List[float]] = None,
    return_stats: bool = False,
    verbose: bool = True,
    log_path: Optional[str] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
    """
    Fit a GP model and optimize an acquisition function to propose next candidate(s).

    Notes:
    - This code assumes minimization (maximize=False) for EI/PI/UCB.
    - `kernel` can be a string (e.g., "rbf", "alas", "alas_sep") or a Module instance.
    """

    dtype = init_x.dtype
    device = init_x.device

    # Likelihood: moderate prior + small floor for stability
    noise_prior = GammaPrior(1.1, 0.5)
    noise_prior_mode = float((noise_prior.concentration - 1) / noise_prior.rate)

    likelihood = GaussianLikelihood(
        noise_prior=noise_prior,
        noise_constraint=GreaterThan(1e-4),
    ).to(device=device, dtype=dtype)

    with torch.no_grad():
        try:
            likelihood.noise = torch.tensor(noise_prior_mode, device=device, dtype=dtype)
        except Exception:
            pass

    model = SingleTaskGP(
        train_X=init_x,
        train_Y=init_y,
        likelihood=likelihood,
        covar_module=kernel,
        n_mixture=n_mixture,
        n_mixture1=n_mixture1,
        n_mixture2=n_mixture2,
        alphas=alphas,
    )

    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    try:
        robust_fit_mll(mll, model, verbose=verbose)
    except Exception as fit_error:
        if verbose:
            print(f"Error during model fitting: {fit_error}")
        model.eval()

    model.eval()
    stats = extract_alpha_snapshot(model)

    if log_path is not None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(stats) + "\n")

    # Acquisition (minimization)
    acq = acq.lower()
    if acq == "ei":
        acqf = ExpectedImprovement(model=model, best_f=best_init_y, maximize=False)
    elif acq == "pi":
        acqf = ProbabilityOfImprovement(model=model, best_f=best_init_y, maximize=False)
    elif acq == "ucb":
        acqf = UpperConfidenceBound(model=model, beta=0.2, maximize=False)
    else:
        raise ValueError(f"Acquisition function '{acq}' not identified")

    candidates, _ = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,
        q=n_points,
        num_restarts=10,
        raw_samples=512,
        options={"batch_limit": 5, "maxiter": 200},
    )

    if verbose:
        cov = model.covar_module
        outputscale = 1.0
        if hasattr(cov, "outputscale"):
            try:
                outputscale = float(cov.outputscale.item())
            except Exception:
                outputscale = 1.0

        noise = float(model.likelihood.noise.item()) if hasattr(model.likelihood, "noise") else float("nan")
        snr = float(outputscale / (noise + 1e-9))
        print(f"  [Model] SNR={snr:.4f} | noise={noise:.6f} | kernel={stats.get('kernel_type')}")

        if stats.get("alpha_bar") is not None:
            print(f"  [Alpha] alpha_bar={stats['alpha_bar']:.4f}")
        else:
            print("  [Alpha] alpha_bar=None")

    if return_stats:
        return candidates, stats
    return candidates


# ===== Optional visualization helpers (used only for 1D inspection) =====

def compute_acquisition_function(single_model: SingleTaskGP, best_init_y: float, l_bound=-2.0, h_bound=10.0, resolution=1000):
    linspace = torch.linspace(
        l_bound,
        h_bound,
        steps=resolution,
        dtype=single_model.train_inputs[0].dtype,
        device=single_model.train_inputs[0].device,
    )
    ei = ExpectedImprovement(model=single_model, best_f=best_init_y, maximize=False)
    with torch.no_grad():
        acq_values = ei(linspace.unsqueeze(-1).unsqueeze(-1))
    return acq_values


def print_acquisition_function(acq_fun, iteration, l_bound=-2.0, h_bound=10.0, resolution=1000, suggested=None):
    if go is None:
        raise ImportError("plotly is not available. Install plotly to use visualization helpers.")

    x_plot = torch.linspace(l_bound, h_bound, steps=resolution).cpu().numpy()
    z_plot = acq_fun.detach().cpu().numpy()
    max_idx = int(acq_fun.argmax().item())
    max_x = float(x_plot[max_idx])

    fig = go.Figure(data=go.Scatter(x=x_plot, y=z_plot, name="Acquisition Function"))
    fig.update_layout(title=f"Acquisition function (iter {iteration})", xaxis_title="x", yaxis_title="acq")

    if suggested is not None:
        fig.add_vline(x=float(suggested[0][0]), line_width=3, line_dash="dash", line_color="red")
    fig.add_vline(x=max_x, line_width=2, line_dash="dot", line_color="orange")
    fig.show()


def compute_predictive_distribution(single_model: SingleTaskGP, l_bound=-2.0, h_bound=10.0, resolution=1000):
    linspace = torch.linspace(
        l_bound,
        h_bound,
        steps=resolution,
        dtype=single_model.train_inputs[0].dtype,
        device=single_model.train_inputs[0].device,
    )
    x_test = linspace.unsqueeze(-1).unsqueeze(-1)
    with torch.no_grad():
        post = single_model.posterior(x_test)
        mean = post.mean.squeeze(-1).squeeze(-1)
        var = post.variance.squeeze(-1).squeeze(-1)
    return mean, var


def print_predictive_mean(predictive_mean, predictive_variance, iteration, l_bound=-2.0, h_bound=10.0, resolution=1000,
                          suggested=None, old_obs=None, old_values=None):
    if go is None:
        raise ImportError("plotly is not available. Install plotly to use visualization helpers.")

    old_obs = old_obs if old_obs is not None else torch.empty(0)
    old_values = old_values if old_values is not None else torch.empty(0)

    x_plot = torch.linspace(l_bound, h_bound, steps=resolution).cpu().numpy()
    mean_plot = predictive_mean.detach().cpu().numpy()
    var_plot = predictive_variance.detach().cpu().numpy()
    std = np.sqrt(np.maximum(var_plot, 1e-12))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_plot, y=mean_plot + 1.96 * std, mode="lines", showlegend=False))
    fig.add_trace(go.Scatter(x=x_plot, y=mean_plot, mode="lines", fill="tonexty", name="Mean + 95% CI"))
    fig.add_trace(go.Scatter(x=x_plot, y=mean_plot - 1.96 * std, mode="lines", fill="tonexty", showlegend=False))

    if old_obs.numel() > 0:
        fig.add_trace(
            go.Scatter(
                x=old_obs.detach().cpu().numpy().flatten(),
                y=old_values.detach().cpu().numpy().flatten(),
                mode="markers",
                name="Observations",
            )
        )

    if suggested is not None:
        fig.add_vline(x=float(suggested[0][0]), line_width=3, line_dash="dash", line_color="red")

    fig.update_layout(title=f"Predictive distribution (iter {iteration})", xaxis_title="x", yaxis_title="y")
    fig.show()


def print_objective_function(target_function, best_candidate, iteration, l_bound=-2.0, h_bound=10.0, resolution=100):
    if go is None:
        raise ImportError("plotly is not available. Install plotly to use visualization helpers.")

    x_plot = np.linspace(l_bound, h_bound, resolution)
    try:
        y_plot = target_function(torch.tensor(x_plot.reshape(-1, 1), dtype=torch.float32)).detach().cpu().numpy().flatten()
    except Exception:
        y_plot = target_function(x_plot.reshape(-1, 1)).flatten()

    fig = go.Figure(data=go.Scatter(x=x_plot, y=y_plot, name="True Objective"))
    fig.update_layout(title=f"Objective (iter {iteration})", xaxis_title="x", yaxis_title="f(x)")

    if best_candidate is not None:
        fig.add_vline(x=float(best_candidate), line_width=2, line_dash="dot", line_color="green")
    fig.show()
