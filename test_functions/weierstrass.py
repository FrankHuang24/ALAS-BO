import numpy as np
import torch
from .optimization_problem import OptimizationProblem


class Weierstrass(OptimizationProblem):
    def __init__(self, dim=10):
        self.dim = dim
        self.min = 0
        self.minimum = np.zeros(dim)
        self.lb = torch.from_numpy(-5 * np.ones(dim)).to(torch.float32)
        self.ub = torch.from_numpy(5 * np.ones(dim)).to(torch.float32)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Weierstrass function"

    def eval(self, x_batch: torch.Tensor) -> torch.Tensor:
        """Evaluate the Weierstrass function for a batch of points.

        :param x_batch: Batch of data points, shape (n, d)
        :type x_batch: torch.Tensor
        :return: Values at x, shape (n, 1)
        :rtype: torch.Tensor
        """
        # BoTorch/GPyTorch works with tensors, so we expect a tensor
        # If it's a numpy array, convert it
        if not isinstance(x_batch, torch.Tensor):
            x_batch = torch.from_numpy(x_batch).to(torch.float32)

        results = []
        for i in range(x_batch.shape[0]):
            x = x_batch[i].numpy() 
    
            d = len(x)
            f0, val = 0.0, 0.0
            for k in range(12):
                f0 += 1.0 / (2 ** k) * np.cos(np.pi * (3 ** k))
                val += (1.0 / (2 ** k)) * np.sum(np.cos(2 * np.pi * (3 ** k) * (x + 0.5)))
            
            result = 10 * ((1.0 / float(d) * val - f0) ** 3)
            results.append(result)
            
        return torch.tensor(results, dtype=torch.float32).unsqueeze(-1)
