"""
This is a surrogate of the portfolio simulator, based on 3k samples found in port_evals.
"""
import math
from typing import Optional
import torch
import gpytorch
from botorch.models.gp_regression_original import SingleTaskGP
from botorch.models.transforms import Standardize
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from torch import Tensor
from .optimization_problem import OptimizationProblem
import os

script_dir = os.path.dirname(os.path.abspath(__file__))


class PortfolioSurrogate(OptimizationProblem):
    r"""
    Surrogate of the portfolio simulator.
    """

    def __init__(self):
        self.min = -12
        self.dim = 5
        self.lb = 0 * torch.ones(5)
        self.ub = 1 * torch.ones(5)
        self.model = None
        self.info = (
                "5-dimensional Portfolio Optimization \nGlobal minimum: "
                + "Unknown"
        )

    def eval(self, X: Tensor) -> Tensor:
        if self.model is not None:
            with torch.no_grad(), gpytorch.settings.max_cg_iterations(10000):
                return self.model.posterior(
                    X.to(dtype=torch.float32, device="cpu")
                ).mean.to(X)
        self.fit_model()
        return -self.eval(X)

    def fit_model(self):
        """
        If no state_dict exists, fits the model and saves the state_dict.
        Otherwise, constructs the model but uses the fit given by the state_dict.
        """
        # read the data
        data_list = list()
        for i in range(1, 31):
            data_file = os.path.join(script_dir, "port_evals", "port_n=100_seed=%d" % i)
            data_list.append(torch.load(data_file))

        # join the data together
        X = torch.cat(
            [data_list[i]["X"] for i in range(len(data_list))], dim=0
        ).squeeze(-2)
        Y = torch.cat(
            [data_list[i]["Y"] for i in range(len(data_list))], dim=0
        ).squeeze(-2)

        # fit GP
        noise_prior = GammaPrior(1.1, 0.5)
        noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
        likelihood = GaussianLikelihood(
            noise_prior=noise_prior,
            batch_shape=[],
            noise_constraint=GreaterThan(
                0.000005,  # minimum observation noise assumed in the GP model
                transform=None,
                initial_value=noise_prior_mode,
            ),
        )

        # We save the state dict to avoid fitting the GP every time which takes ~3 mins
        try:
            state_dict = torch.load(
                os.path.join(script_dir, "portfolio_surrogate_state_dict.pt")
            )
            model = SingleTaskGP(train_X=X, train_Y=Y, likelihood=likelihood, outcome_transform=Standardize(m=1))
            model.load_state_dict(state_dict)
        except FileNotFoundError:
            model = SingleTaskGP(train_X=X, train_Y=Y, likelihood=likelihood, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            from time import time

            start = time()
            fit_gpytorch_mll(mll)
            print("fitting took %s seconds" % (time() - start))
            torch.save(
                model.state_dict(),
                os.path.join(script_dir, "portfolio_surrogate_state_dict.pt"),
            )
        self.model = model
