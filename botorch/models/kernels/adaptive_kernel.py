#!/usr/bin/env python3

from typing import Optional, Union, List, Type

import torch
from linear_operator.operators import LinearOperator, MatmulLinearOperator, RootLinearOperator
from torch import Tensor

import gpytorch
from gpytorch.constraints import Interval, Positive
from gpytorch.priors import Prior
from gpytorch.kernels.kernel import Kernel



class AdaptiveKernel(Kernel):
    r"""
    Computes a covariance matrix by adaptively selecting the best kernel from a list of candidates
    based on marginal log likelihood (MLL) during training.

    Args:
        kernel_list: List of kernel classes or instances to choose from.
        ard_num_dims: Set this if you want a separate lengthscale for each input dimension.
        variance_prior: Prior over the variance parameter.
        variance_constraint: Constraint to place on variance parameter.
        active_dims: List of data dimensions to operate on.
        **kwargs: Additional arguments to pass to each kernel initialization.
    """

    def __init__(
            self,
            kernel_list: List[Union[Type[Kernel], Kernel]],
            ard_num_dims: Optional[int] = None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.kernels = torch.nn.ModuleList()
        for kernel in kernel_list:
            if isinstance(kernel, Kernel):
                self.kernels.append(kernel)  # Instance
            else:
                self.kernels.append(kernel(  # Initialization from Class
                    ard_num_dims=ard_num_dims,
                    **kwargs
                ))

        self.active_kernel_idx = None
        self._prev_mlls = None

    def select_best_kernel(self, x: Tensor, y: Tensor) -> int:
        """
        Selects the kernel with highest marginal log likelihood given current data.

        Args:
            x: Input training data.
            y: Target training data.

        Returns:
            Index of the best kernel.
        """
        mlls = []
        for kernel in self.kernels:
            # Compute marginal log likelihood for each kernel
            K = kernel(x, x).add_jitter(1e-6)
            try:
                L = K.cholesky()
                mll = gpytorch.distributions.MultivariateNormal(
                    torch.zeros_like(y), L
                ).log_prob(y)
                mlls.append(mll.item())
            except:
                mlls.append(-float('inf'))

        self._prev_mlls = mlls
        return torch.argmax(torch.tensor(mlls)).item()

    @property
    def active_kernel(self) -> Kernel:
        """Returns the currently active kernel."""
        if self.active_kernel_idx is None:
            raise RuntimeError("No active kernel selected. Call forward() first.")
        return self.kernels[self.active_kernel_idx]

    def forward(
            self,
            x1: Tensor,
            x2: Tensor,
            diag: bool = False,
            last_dim_is_batch: bool = False,
            select_kernel: bool = True,
            **params
    ) -> Union[Tensor, LinearOperator]:
        """
        Forward pass that automatically selects best kernel on first call.

        Args:
            select_kernel: If True, will select best kernel before forward pass.
                          Set to False after first call for consistent behavior.
        """
        if select_kernel and (self.active_kernel_idx is None or self.training):
            # During training or first call, select best kernel
            if not hasattr(self, "_train_x") or not hasattr(self, "_train_y"):
                raise RuntimeError("Must set train data before forward pass")
            self.active_kernel_idx = self.select_best_kernel(self._train_x, self._train_y)

        return self.active_kernel(x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)

    def set_train_data(self, x: Tensor, y: Tensor):
        """Store training data for kernel selection."""
        self._train_x = x
        self._train_y = y

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"(kernels={[k.__class__.__name__ for k in self.kernels]})"