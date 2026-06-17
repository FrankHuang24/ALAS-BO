import torch
from .optimization_problem import OptimizationProblem
from .hartmann6 import Hartmann6 

class Hartmann24(OptimizationProblem):
    """
    24-dimensional Hartmann function, constructed by additively composing
    four independent 6-dimensional Hartmann functions.
    f(x_1,...,x_24) = H6(x_1..6) + H6(x_7..12) + H6(x_13..18) + H6(x_19..24)
    
    This function is designed to test the performance of additive models in
    high dimensions.
    """

    def __init__(self):
        self.dim = 24
        
        h6 = Hartmann6() 
        self.min = 4 * h6.min
        self.minimum = torch.cat([h6.minimum] * 4)
        self.lb = torch.cat([h6.lb] * 4)
        self.ub = torch.cat([h6.ub] * 4)

        self.int_var = torch.tensor([], dtype=torch.long)
        self.cont_var = torch.arange(0, 24)
        self.info = "24-dimensional Additive Hartmann function"

        self.h6_modules = [Hartmann6() for _ in range(4)]

    def eval(self, x_batch: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the Additive Hartmann 24 function for a batch of points.
        """
        self.__check_input__(x_batch)
        if x_batch.shape[1] != self.dim:
            raise ValueError(f"Input dimension must be {self.dim}")

        results = [
            self.h6_modules[0].eval(x_batch[:, 0:6]),
            self.h6_modules[1].eval(x_batch[:, 6:12]),
            self.h6_modules[2].eval(x_batch[:, 12:18]),
            self.h6_modules[3].eval(x_batch[:, 18:24]),
        ]

        return torch.sum(torch.cat(results, dim=1), dim=1).unsqueeze(-1)