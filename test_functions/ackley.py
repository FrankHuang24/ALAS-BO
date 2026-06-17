import torch
import numpy as np
from .optimization_problem import OptimizationProblem

class Ackley(OptimizationProblem):
    """Ackley function implemented in PyTorch
    
    Supports batch evaluation and high-precision (float64).
    """

    def __init__(self, dim=10):
        self.dim = dim
        self.min = 0.0
        self.minimum = torch.zeros(dim, dtype=torch.float64)
        
        self.lb = -10.0 * torch.ones(dim, dtype=torch.float64)
        self.ub =  10.0 * torch.ones(dim, dtype=torch.float64)
        
        self.int_var = torch.tensor([], dtype=torch.int32)
        self.cont_var = torch.arange(0, dim)
        self.info = str(dim) + "-dimensional Ackley function \n" + "Global optimum: f(0,0,...,0) = 0"

    def eval(self, x):
        """Evaluate the Ackley function at x

        :param x: Data point
        :type x: torch.Tensor of shape (n, dim)
        :return: Value at x
        :rtype: torch.Tensor of shape (n, 1)
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float64)
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        d = float(self.dim)
        
        # term 1: exp( -0.2 * sqrt( sum(x^2)/d ) )
        sum_sq = torch.sum(x ** 2, dim=1)
        term1 = -20.0 * torch.exp(-0.2 * torch.sqrt(sum_sq / d))
        
        # term 2: exp( sum(cos(2pi*x))/d )
        sum_cos = torch.sum(torch.cos(2.0 * torch.pi * x), dim=1)
        term2 = -torch.exp(sum_cos / d)
        
        f = term1 + term2 + 20.0 + torch.exp(torch.tensor(1.0, dtype=x.dtype, device=x.device))

        return f.unsqueeze(1)