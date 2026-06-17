# rastrigin.py

import numpy as np
import torch

from .optimization_problem import OptimizationProblem


class Rastrigin(OptimizationProblem):
    """ """

    def __init__(self, dim=10):
        self.dim = dim
        self.min = 0
        self.minimum = np.zeros(dim)
        
        self.lb = torch.from_numpy(-5.12 * np.ones(dim)).to(torch.float32)
        self.ub = torch.from_numpy(5.12 * np.ones(dim)).to(torch.float32)
        
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Rastrigin function \n" + "Global optimum: f(0,0,...,0) = 0"

    def eval(self, x_batch: torch.Tensor) -> torch.Tensor:
        """Evaluate the Rastrigin function for a batch of points.

        :param x_batch: Batch of data points, shape (n, d)
        :type x_batch: torch.Tensor
        :return: Values at x, shape (n, 1)
        :rtype: torch.Tensor
        """
        if not isinstance(x_batch, torch.Tensor):
            x_batch = torch.from_numpy(x_batch).to(torch.float32)
        
        sum_term = torch.sum(x_batch**2 - 10 * torch.cos(2 * torch.pi * x_batch), dim=1)
        
        result = 10 * self.dim + sum_term
        
        return result.unsqueeze(-1)
