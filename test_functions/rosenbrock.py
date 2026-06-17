import torch
import numpy as np
from .optimization_problem import OptimizationProblem

class Rosenbrock(OptimizationProblem):
    """Rosenbrock benchmark implemented with PyTorch.

    Domain: [-2.048, 2.048]^dim.
    Global optimum: f(1, ..., 1) = 0.
    """

    def __init__(self, dim=14):
        self.dim = dim
        self.min = 0
        self.minimum = torch.ones(dim)
        self.lb = -2.048 * torch.ones(dim)
        self.ub = 2.048 * torch.ones(dim)
        self.int_var = torch.tensor([])  # No integer variables
        self.cont_var = torch.arange(0, dim)  # Continuous variables
        self.info = f"{dim}-dimensional Rosenbrock function \n" + "Global optimum: f(1,1,...,1) = 0"

    def eval(self, x):
        """Evaluate the Rosenbrock function for each row of x."""

        total = 0.0
        for i in range(self.dim - 1):
            total += 100 * (x[:, i]**2 - x[:, i + 1])**2 + (1 - x[:, i])**2

        return total[:, None]
