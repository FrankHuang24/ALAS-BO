import torch
import math
from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive, Interval
from gpytorch.priors import Prior
from typing import Optional, Union


class SincKernel(Kernel):
    r"""
    Implements the Sinc kernel from "Band-Limited Gaussian Processes: The Sinc Kernel" (Tobar 2019).

    The kernel is defined as:
    K(t) = variance * sinc(bandwidth * t) * cos(2 * pi * center_freq * t)

    where:
    - sinc(x) = sin(πx)/(πx)
    - bandwidth controls the frequency band width (Δ in paper)
    - center_freq controls the central frequency (ξ₀ in paper)
    - variance controls the overall scale (σ² in paper)
    """

    def __init__(
            self,
            bandwidth: float = 1.0,
            center_freq: float = 0.5,
            variance: float = 1.0,
            bandwidth_prior: Optional[Prior] = None,
            bandwidth_constraint: Optional[Interval] = None,
            center_freq_prior: Optional[Prior] = None,
            center_freq_constraint: Optional[Interval] = None,
            variance_prior: Optional[Prior] = None,
            variance_constraint: Optional[Interval] = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        # Set default parameter constraints if not provided
        if bandwidth_constraint is None:
            bandwidth_constraint = Interval(1e-3, 1e3)  # Prevents numerical instability
        if center_freq_constraint is None:
            center_freq_constraint = Positive()  # Ensures center frequency is non-negative
        if variance_constraint is None:
            variance_constraint = Positive()  # Ensures variance is positive

        # Register learnable parameters
        self.register_parameter("raw_bandwidth", torch.nn.Parameter(torch.tensor(float(bandwidth))))
        self.register_parameter("raw_center_freq", torch.nn.Parameter(torch.tensor(float(center_freq))))
        self.register_parameter("raw_variance", torch.nn.Parameter(torch.tensor(float(variance))))

        # Apply constraints
        self.register_constraint("raw_bandwidth", bandwidth_constraint)
        self.register_constraint("raw_center_freq", center_freq_constraint)
        self.register_constraint("raw_variance", variance_constraint)

        # Register priors (if provided)
        if bandwidth_prior is not None:
            self.register_prior("bandwidth_prior", bandwidth_prior, lambda: self.bandwidth,
                                lambda v: self._set_bandwidth(v))
        if center_freq_prior is not None:
            self.register_prior("center_freq_prior", center_freq_prior, lambda: self.center_freq,
                                lambda v: self._set_center_freq(v))
        if variance_prior is not None:
            self.register_prior("variance_prior", variance_prior, lambda: self.variance,
                                lambda v: self._set_variance(v))

    # Property accessors for kernel parameters
    @property
    def bandwidth(self) -> torch.Tensor:
        return self.raw_bandwidth_constraint.transform(self.raw_bandwidth)

    def _set_bandwidth(self, value: Union[float, torch.Tensor]) -> None:
        if not torch.is_tensor(value):
            value = torch.tensor(value)
        self.initialize(raw_bandwidth=self.raw_bandwidth_constraint.inverse_transform(value))

    @property
    def center_freq(self) -> torch.Tensor:
        return self.raw_center_freq_constraint.transform(self.raw_center_freq)

    def _set_center_freq(self, value: Union[float, torch.Tensor]) -> None:
        if not torch.is_tensor(value):
            value = torch.tensor(value)
        self.initialize(raw_center_freq=self.raw_center_freq_constraint.inverse_transform(value))

    @property
    def variance(self) -> torch.Tensor:
        return self.raw_variance_constraint.transform(self.raw_variance)

    def _set_variance(self, value: Union[float, torch.Tensor]) -> None:
        if not torch.is_tensor(value):
            value = torch.tensor(value)
        self.initialize(raw_variance=self.raw_variance_constraint.inverse_transform(value))

    def forward(
            self,
            x1: torch.Tensor,
            x2: torch.Tensor,
            diag: bool = False,
            last_dim_is_batch: bool = False,
            **params
    ) -> torch.Tensor:
        """
        Compute the kernel matrix between x1 and x2.

        Args:
            x1 (torch.Tensor): Input tensor of shape (batch_size, N, d)
            x2 (torch.Tensor): Input tensor of shape (batch_size, M, d)
            diag (bool): If True, return only the diagonal of the kernel matrix.
            last_dim_is_batch (bool): Unused, but required for compatibility.

        Returns:
            torch.Tensor: Kernel matrix of shape (batch_size, N, M)
        """

        # Ensure correct input shape (if 1D, add extra dimension)
        if x1.ndim == 1:
            x1 = x1.unsqueeze(-1)
        if x2.ndim == 1:
            x2 = x2.unsqueeze(-1)

        assert x1.size(-1) == x2.size(-1), "Feature dimension mismatch"

        # Compute pairwise L1 distance to ensure shape consistency
        t = torch.cdist(x1, x2, p=1)  # Shape: (batch_size, N, M)

        # Ensure numerical stability by clamping bandwidth away from zero
        bandwidth_t = self.bandwidth.clamp(min=1e-6) * t

        # Compute sinc function with numerical stability for small values
        sinc_val = torch.where(
            bandwidth_t < 1e-4,
            1 - (math.pi ** 2 * bandwidth_t ** 2) / 6,  # Taylor series approximation for sinc(x) near 0
            torch.sinc(bandwidth_t / math.pi)
        )

        # Compute cosine modulation term
        cos_val = torch.cos(2 * math.pi * self.center_freq * t)

        # Compute final kernel matrix
        K = self.variance * sinc_val * cos_val

        # Add small jitter for numerical stability when x1 == x2
        if not diag and torch.equal(x1, x2):
            K = K + torch.eye(K.size(-1), device=K.device) * 1e-6

        return K.diagonal(dim1=-2, dim2=-1) if diag else K

    def __repr__(self) -> str:
        return (
            f"SincKernel(bandwidth={self.bandwidth:.3g}, "
            f"center_freq={self.center_freq:.3g}, "
            f"variance={self.variance:.3g})"
        )
