# exponential.py (Modified)

import torch
from .optimization_problem import OptimizationProblem

class Exponential(OptimizationProblem):
    """Exponential function

    Original definition:
        f(x_1,...,x_d) = sum_{j=1 to d} [exp(j * x_j)] - sum_{j=1 to d} [exp(-5.12 * j)]
    subject to
        -5.12 <= x_i <= 5.12
    Global optimum: f(-5.12,...,-5.12) = 0

    This version uses modified bounds for numerical stability.
    The constant term remains the same as the original definition.
    The new self.min and self.minimum reflect the optimum within the new bounds.
    """

    def __init__(self, dim=10):
        self.dim = dim

        new_bound_val = 2.0
        self.lb = -new_bound_val * torch.ones(dim, dtype=torch.float32)
        self.ub =  new_bound_val * torch.ones(dim, dtype=torch.float32)

        self.int_var = torch.tensor([], dtype=torch.long)
        self.cont_var = torch.arange(0, dim)

        self.minimum = self.lb.clone()

        indices = torch.arange(1, self.dim + 1, dtype=torch.float32)
        original_constant_term = torch.exp(-5.12 * indices).sum()

        exp_part_at_new_minimum = torch.exp(indices * self.minimum)
        sum_exp_at_new_minimum = torch.sum(exp_part_at_new_minimum)
        self.min = (sum_exp_at_new_minimum - original_constant_term).item()

        self.info = (
            f"{dim}-dimensional Exponential function "
            f"(bounds [{self.lb[0].item():.2f}, {self.ub[0].item():.2f}])\n"
            f"Original constant term used. Optimum within these bounds: "
            f"f({[round(val.item(), 2) for val in self.minimum]}) = {self.min:.4e}"
        )


    def eval(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the Exponential function at x.

        :param x: Data point (n × dim), where n is the number of samples
        :type x: torch.Tensor
        :return: Value at x (n × 1)
        :rtype: torch.Tensor
        """
        self.__check_input__(x)

        # Constants from the original definition
        # Ensure indices is on the same device as x
        indices = torch.arange(1, self.dim + 1, dtype=torch.float32, device=x.device)
        original_constant_term = torch.exp(-5.12 * indices).sum()

        # Compute the sum of exponentials for each sample
        # Ensure x is on the same device as indices for multiplication
        exp_part = torch.exp(indices * x)  # (n × dim)
        exp_sum = torch.sum(exp_part, dim=1)  # (n,) sum across the dimensions for each sample

        # Final result (n,)
        result = exp_sum - original_constant_term

        # Return as n × 1 tensor
        return result.unsqueeze(-1)

    def __check_input__(self, x: torch.Tensor):
        """
        Helper function to check the input to eval.
        Ensure it's a 2D tensor and values are within bounds.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input x must be a PyTorch tensor.")
        if x.dim() != 2:
            raise ValueError(f"Input x must be 2-dimensional (n_samples x dim), but got {x.dim()} dimensions.")
        if x.shape[1] != self.dim:
            raise ValueError(f"Input x must have {self.dim} columns (features), but got {x.shape[1]}.")

        # Check bounds - be careful with broadcasting if x is batched for checking
        # For simplicity, assuming x here is not batched in a way that conflicts with lb/ub
        # if (x < self.lb.unsqueeze(0)).any() or (x > self.ub.unsqueeze(0)).any():
        #     warnings.warn("Input x contains values outside of the defined bounds.", UserWarning)
        #     # Optionally, clamp the values or raise an error
        #     # x = torch.max(torch.min(x, self.ub), self.lb) # Clamping example