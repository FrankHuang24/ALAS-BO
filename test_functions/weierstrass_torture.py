import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Iterable, Tuple, Optional

@dataclass
class WeierstrassTorture:
    """
    A torture test for BO: Heavy tails + Non-smoothness (Cusps) + Multiscale.
    Fully compatible with the run_stable.py interface.
    """
    dim: int = 8
    # Weierstrass parameters
    a: float = 0.5            
    b: float = 3.0            
    K: int = 20               
    
    # Torture parameters
    cusp_dims: Iterable[int] = (0, 1)
    cusp_weight: float = 0.25
    lowfreq_weight: float = 0.2
    seed: int = 0
    
    # Interface compliance
    min: float = 0.0 # Unknown global minimum, placeholder for regret calc

    def __post_init__(self):
        # 1. Setup Randomness
        rng = np.random.default_rng(self.seed)
        
        # 2. Internal parameters (NumPy)
        self._cusp_dims = tuple(sorted(set(int(i) for i in self.cusp_dims if 0 <= i < self.dim)))
        self._phases = rng.uniform(0.0, 1.0, size=self.dim)
        self._cusp_centers = rng.uniform(0.2, 0.8, size=self.dim)
        self._per_dim_amp = 0.6 + 0.8 * rng.random(self.dim)
        self._lowfreq = rng.integers(1, 3, size=self.dim)
        
        # 3. Run_stable Interface: Bounds as Tensors
        # Define domain as [-0.5, 0.5] to center the complexity, or [0, 1]
        # Let's stick to [0, 1] as in your original design
        self.lb = torch.zeros(self.dim, dtype=torch.double)
        self.ub = torch.ones(self.dim, dtype=torch.double)

    def _weierstrass_1d_np(self, x: np.ndarray, j: int) -> np.ndarray:
        """Vectorized 1D Weierstrass in NumPy."""
        x_shift = x + self._phases[j]
        ks = np.arange(self.K + 1)
        # shape: (N, 1) * (K+1,) -> (N, K+1)
        term = (self.a ** ks) * np.cos(2.0 * np.pi * (self.b ** ks) * x_shift[..., None])
        return term.sum(axis=-1)

    def _evaluate_numpy(self, x: np.ndarray) -> np.ndarray:
        """Internal calculation using optimized NumPy vectorization."""
        single = x.ndim == 1
        if single: x = x[None, :]
        
        # 1. Weierstrass high-freq part
        wsum = np.zeros(x.shape[0], dtype=float)
        for j in range(self.dim):
            wsum += self._per_dim_amp[j] * self._weierstrass_1d_np(x[:, j], j)

        # 2. Cusp part (Non-differentiable)
        if self._cusp_dims:
            for j in self._cusp_dims:
                wsum += self.cusp_weight * np.abs(x[:, j] - self._cusp_centers[j])

        # 3. Low-freq trend
        lr = np.zeros(x.shape[0], dtype=float)
        for j in range(self.dim):
            lr += self.lowfreq_weight * np.cos(2.0 * np.pi * self._lowfreq[j] * x[:, j])

        # 4. Normalization (Scale by sqrt(dim) to keep range reasonable)
        y = (wsum + lr) / np.sqrt(self.dim)
        
        return y[0] if single else y

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        The main interface for run_stable.py.
        Handles Tensor <-> NumPy conversion seamlessly.
        """
        # 1. Input validation & conversion
        original_device = x.device
        original_dtype = x.dtype
        
        # Detach and move to CPU NumPy
        if x.dim() == 1:
            x_np = x.detach().cpu().view(1, -1).numpy()
        else:
            x_np = x.detach().cpu().numpy()
            
        # 2. Calculate
        y_np = self._evaluate_numpy(x_np)
        
        # 3. Convert back to Tensor
        y_tensor = torch.tensor(y_np, device=original_device, dtype=original_dtype)
        
        # Ensure shape is (N, 1) if input was (N, D)
        if y_tensor.dim() == 1:
            y_tensor = y_tensor.unsqueeze(-1)
            
        return y_tensor