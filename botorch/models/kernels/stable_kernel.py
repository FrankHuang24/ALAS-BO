"""
ALAS kernel implementations.

This module contains the exact-GP kernels used by the paper:
- LearnableAlphaStableKernel: coupled ALAS kernel.
- AdditiveLearnableAlphaStableKernel: ALAS-Sep additive kernel.
"""


import math
from typing import List, Optional, Union
import numpy as np
from scipy.fftpack import fft
import torch
from gpytorch.constraints import Interval, Positive,GreaterThan
from gpytorch.priors import Prior, LogNormalPrior, NormalPrior, GammaPrior
from gpytorch.kernels.kernel import Kernel

class LearnableAlphaStableKernel(Kernel):
    r"""
    Learnable alpha-stable kernel with optional mixture components.
    """
    is_stationary = True
    has_lengthscale = False

    def __init__(
        self,
        num_mixtures: int, 
        num_dims: int,    
        batch_shape: Optional[torch.Size] = torch.Size([]),
        alphas_prior: Optional[Prior] = None,
        alphas_constraint: Optional[Interval] = None,
        mixture_weights_prior: Optional[Prior] = None,
        mixture_weights_constraint: Optional[Interval] = None,
        mixture_scales_prior: Optional[Prior] = None,
        mixture_scales_constraint: Optional[Interval] = None,
        mixture_means_prior: Optional[Prior] = None,
        mixture_means_constraint: Optional[Interval] = None,
        active_dims: Optional[List[int]] = None,
        initial_dt_override: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            batch_shape=batch_shape,
            active_dims=active_dims,
            ard_num_dims=num_dims,
            **kwargs
        )
        self.num_mixtures = num_mixtures
        self.fixed_dt_for_run = initial_dt_override
        
        param_shape_ard = list(batch_shape) + [self.num_mixtures, 1, num_dims]
        param_shape_scalar = list(batch_shape) + [self.num_mixtures]

        # Stability parameters.
        self.register_parameter(name="raw_alphas", parameter=torch.nn.Parameter(torch.zeros(param_shape_scalar)))
        if alphas_constraint is None: alphas_constraint = Interval(0.01, 2.0)
        self.register_constraint("raw_alphas", alphas_constraint)
        if alphas_prior is not None: self.register_prior("alphas_prior", alphas_prior, lambda m: m.alphas, lambda m, v: m._set_alphas(v))
        if self.num_mixtures > 1:
            initial_alphas_1d = torch.linspace(1.0, 2.0, self.num_mixtures)
        else:
            initial_alphas_1d = torch.tensor([1.5])
        view_shape = [1] * len(batch_shape) + [self.num_mixtures]
        initial_alphas = initial_alphas_1d.view(*view_shape).expand(*param_shape_scalar)
        self.alphas = initial_alphas

        # Mixture weights.
        self.register_parameter(name="raw_mixture_weights", parameter=torch.nn.Parameter(torch.zeros(param_shape_scalar)))
        if mixture_weights_constraint is None: mixture_weights_constraint = Positive()
        self.register_constraint("raw_mixture_weights", mixture_weights_constraint)
        if mixture_weights_prior is not None: self.register_prior("mixture_weights_prior", mixture_weights_prior, self._get_mixture_weights, self._set_mixture_weights_unnormalized)
        self.mixture_weights = torch.ones(param_shape_scalar)

        # Spectral scales.
        self.register_parameter(name="raw_mixture_scales", parameter=torch.nn.Parameter(torch.zeros(param_shape_ard)))
        if mixture_scales_constraint is None: mixture_scales_constraint = GreaterThan(1e-6)
        self.register_constraint("raw_mixture_scales", mixture_scales_constraint)
        if mixture_scales_prior is None: mixture_scales_prior = LogNormalPrior(0.0, 1.0)
        if mixture_scales_prior is not None: 
            self.register_prior(
                "mixture_scales_prior", 
                mixture_scales_prior, 
                "mixture_scales" 
            )

        # Spectral modulation frequencies.
        self.register_parameter(name="raw_mixture_means", parameter=torch.nn.Parameter(torch.zeros(param_shape_ard)))
        if mixture_means_constraint is not None: self.register_constraint("raw_mixture_means", mixture_means_constraint)
        if mixture_means_prior is None: mixture_means_prior = NormalPrior(0.0, 1.0)
        if mixture_means_prior is not None:
            self.register_prior(
                "mixture_means_prior", 
                mixture_means_prior, 
                lambda m: m.mixture_means, 
                lambda m, v: setattr(m, 'mixture_means', v)
            )
        self.mixture_means = torch.randn(param_shape_ard) * 0.01

    @property
    def alphas(self): return self.raw_alphas_constraint.transform(self.raw_alphas)
    @alphas.setter
    def alphas(self, value):
        if not torch.is_tensor(value): value = torch.as_tensor(value).to(self.raw_alphas)
        self.initialize(raw_alphas=self.raw_alphas_constraint.inverse_transform(value))
    def _set_alphas(self, value): self.alphas = value

    @property
    def mixture_weights(self):
        raw_val = self.raw_mixture_weights_constraint.transform(self.raw_mixture_weights)
        return raw_val / (raw_val.sum(dim=-1, keepdim=True) + 1e-8)
    @mixture_weights.setter
    def mixture_weights(self, value):
        if not torch.is_tensor(value): value = torch.as_tensor(value).to(self.raw_mixture_weights)
        self.initialize(raw_mixture_weights=self.raw_mixture_weights_constraint.inverse_transform(value))
    def _get_mixture_weights(self, m): return m.mixture_weights
    def _set_mixture_weights_unnormalized(self, value): self.mixture_weights = value
    
    @property
    def mixture_scales(self): return self.raw_mixture_scales_constraint.transform(self.raw_mixture_scales)
    @mixture_scales.setter
    def mixture_scales(self, value):
        if not torch.is_tensor(value): value = torch.as_tensor(value).to(self.raw_mixture_scales)
        self.initialize(raw_mixture_scales=self.raw_mixture_scales_constraint.inverse_transform(value))

    @property
    def mixture_means(self):
        constraint = getattr(self, "raw_mixture_means_constraint", None)
        return constraint.transform(self.raw_mixture_means) if constraint is not None else self.raw_mixture_means
    @mixture_means.setter
    def mixture_means(self, value):
        if not torch.is_tensor(value): value = torch.as_tensor(value).to(self.raw_mixture_means)
        constraint = getattr(self, "raw_mixture_means_constraint", None)
        if constraint is not None: self.initialize(raw_mixture_means=constraint.inverse_transform(value))
        else: self.initialize(raw_mixture_means=value)
    def forward(self, x1, x2, diag=False, **params):
        if x1.ndim == 1: x1 = x1.unsqueeze(-1)
        if x2.ndim == 1: x2 = x2.unsqueeze(-1)
        if self.ard_num_dims is not None:
             if x1.shape[-1] != self.ard_num_dims: raise RuntimeError(f"x1 expected {self.ard_num_dims} dimensions, got {x1.shape[-1]}")
             if x2.shape[-1] != self.ard_num_dims: raise RuntimeError(f"x2 expected {self.ard_num_dims} dimensions, got {x2.shape[-1]}")
        if diag:
            tau = torch.zeros(list(x1.shape[:-1]), dtype=x1.dtype, device=x1.device)
            diff_d = torch.zeros_like(x1)
        else:
            diff_d = x1.unsqueeze(-2) - x2.unsqueeze(-3)
            tau = diff_d.norm(dim=-1)
        weights = self.mixture_weights
        deltas = self.mixture_scales
        gammas = self.mixture_means
        alphas = self.alphas
        output_shape = list(x1.shape[:-2]) + list(tau.shape[-2:])
        kernel_res = torch.zeros(output_shape, dtype=x1.dtype, device=x1.device)
        for i in range(self.num_mixtures):
            alpha_i = alphas[..., i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            weight_view_shape = list(weights.shape[:-1]) + [1] * tau.dim()
            w_i = weights[..., i].view(weight_view_shape)
            delta_i = deltas[..., i, :, :]
            gamma_i = gammas[..., i, :, :]
            delta_i_broadcast = delta_i.unsqueeze(-2) if not diag else delta_i
            gamma_i_broadcast = gamma_i.unsqueeze(-2) if not diag else gamma_i
            abs_diff_d = diff_d.abs()
            epsilon = 1e-8
            term1_input = 2 * math.pi * delta_i_broadcast * abs_diff_d
            if torch.any(alphas[..., i] < 1.0):
                term1_input = torch.where(term1_input < epsilon, term1_input + epsilon, term1_input)
            term1_base = term1_input.pow(alpha_i.squeeze())
            exp_arg = term1_base.sum(dim=-1)
            term2 = 2 * math.pi * gamma_i_broadcast * diff_d
            cos_arg = term2.sum(dim=-1)
            kernel_i = w_i * torch.exp(-exp_arg) * torch.cos(cos_arg)
            kernel_res = kernel_res + kernel_i
        return kernel_res
    
    def initialize_from_data_empspect(self, train_x: torch.Tensor, train_y: torch.Tensor):
        if not torch.is_tensor(train_x) or not torch.is_tensor(train_y):
            raise RuntimeError("train_x and train_y should be tensors")

        if train_x.dim() < 3: train_x = train_x.unsqueeze(0)
        if train_y.dim() < 3:
             if train_y.dim() == 1: train_y = train_y.unsqueeze(0).unsqueeze(-1)
             elif train_y.dim() == 2: train_y = train_y.unsqueeze(0)
        if train_y.shape[0] != train_x.shape[0]:
             train_y = train_y.expand(train_x.shape[0], *train_y.shape[1:])

        with torch.no_grad():
            train_x_detached = train_x.detach()
            train_y_detached = train_y.detach()

            if train_y_detached.shape[-1] > 1:
                y_for_fft = train_y_detached.mean(dim=-1)
            else:
                y_for_fft = train_y_detached.squeeze(-1)

            batch_size = y_for_fft.shape[0]
            all_batch_means, all_batch_scales, all_batch_weights = [], [], []
            
            for b in range(batch_size):
                current_y = y_for_fft[b].cpu().numpy()
                N = len(current_y)
                
                def fallback_init():
                    print(f"  Fallback: Using random initialization for batch {b}.")
                    rand_means = torch.rand(self.num_mixtures, 1, self.ard_num_dims, device=self.raw_mixture_means.device, dtype=self.raw_mixture_means.dtype) * 0.5
                    rand_scales = torch.rand(self.num_mixtures, 1, self.ard_num_dims, device=self.raw_mixture_scales.device, dtype=self.raw_mixture_scales.dtype) * 0.5 + 0.1
                    rand_weights = torch.ones(self.num_mixtures, device=self.raw_mixture_weights.device, dtype=self.raw_mixture_weights.dtype) / self.num_mixtures
                    all_batch_means.append(rand_means); all_batch_scales.append(rand_scales); all_batch_weights.append(rand_weights)

                if N < 2:
                    print(f"Warning: Not enough data points ({N}) in batch {b} for FFT.")
                    fallback_init(); continue

                emp_spect_full = np.abs(fft(current_y))**2 / N
                dt = self.fixed_dt_for_run if self.fixed_dt_for_run is not None else 1.0
                freq = np.fft.rfftfreq(N, d=dt)
                emp_spect = emp_spect_full[:len(freq)]

                if len(emp_spect) < 1:
                    print(f"Warning: Zero frequency components after rfftfreq for batch {b}.")
                    fallback_init(); continue

                num_peaks_to_find = self.num_mixtures
                if len(emp_spect) < self.num_mixtures:
                    print(f"Warning: Not enough frequency components ({len(emp_spect)}) to initialize {self.num_mixtures} mixtures. Using all {len(emp_spect)} components.")
                    num_peaks_to_find = len(emp_spect)
                
                if num_peaks_to_find == 0:
                    print(f"Warning: num_peaks_to_find is 0 for batch {b}. Using fallback init.")
                    fallback_init(); continue

                peak_indices = np.argsort(emp_spect)[-num_peaks_to_find:]
                init_means = freq[peak_indices]
                max_emp_spect = np.max(emp_spect); max_emp_spect = max_emp_spect if max_emp_spect > 1e-8 else 1e-8
                init_scales = np.sqrt(emp_spect[peak_indices] / max_emp_spect) * 0.5; init_scales = np.abs(init_scales) + 1e-6
                init_weights = emp_spect[peak_indices]
                sum_init_weights = np.sum(init_weights)
                if sum_init_weights > 1e-8: init_weights /= sum_init_weights
                else: init_weights = np.ones(num_peaks_found) / num_peaks_found if num_peaks_found > 0 else np.array([]) 

                if len(init_means) < self.num_mixtures:
                    num_pad = self.num_mixtures - len(init_means)
                    print(f"  Padding with {num_pad} default component(s).")
        
                    if len(init_means) > 0:
                        mean_range = (np.min(init_means), np.max(init_means))
                    else: 
                        mean_range = (0.01, 0.1)
                    pad_means = np.random.uniform(mean_range[0], mean_range[1] + 0.1, num_pad)
                    init_means = np.concatenate([init_means, pad_means])
                    
                    if len(init_scales) > 0:
                        median_scale = np.median(init_scales)
                    else:
                        median_scale = 0.5
                    pad_scales = np.full(num_pad, median_scale)
                    init_scales = np.concatenate([init_scales, pad_scales])
                    
                    if len(init_weights) > 0:
                        pad_weights = np.full(num_pad, np.mean(init_weights) * 0.1 + 1e-6)
                    else:
                        pad_weights = np.full(num_pad, 1.0 / self.num_mixtures)
                    init_weights = np.concatenate([init_weights, pad_weights])
                    init_weights /= np.sum(init_weights) 
                if len(init_means) != self.num_mixtures:
                    print(f"Error: After padding, parameter lengths ({len(init_means)}) do not match num_mixtures ({self.num_mixtures}).")
                    fallback_init() 
                    continue      

                dtype = self.raw_mixture_means.dtype; device = self.raw_mixture_means.device
                batch_init_means = torch.tensor(init_means, dtype=dtype, device=device).unsqueeze(-1).unsqueeze(-1).expand(self.num_mixtures, 1, self.ard_num_dims)
                all_batch_means.append(batch_init_means)
                batch_init_scales = torch.tensor(init_scales, dtype=dtype, device=device).unsqueeze(-1).unsqueeze(-1).expand(self.num_mixtures, 1, self.ard_num_dims)
                all_batch_scales.append(batch_init_scales)
                batch_init_weights = torch.tensor(init_weights, dtype=dtype, device=device)
                all_batch_weights.append(batch_init_weights)

            if all_batch_means and len(all_batch_means) == batch_size:
                self.mixture_means = torch.stack(all_batch_means, dim=0)
                self.mixture_scales = torch.stack(all_batch_scales, dim=0)
                self.mixture_weights = torch.stack(all_batch_weights, dim=0)
            else:
                print(f"Warning: Empirical spectrum initialization failed for {self.__class__.__name__}. Using default/random init.")

class AdditiveLearnableAlphaStableKernel(Kernel):
    r"""
    Additive ALAS-Sep kernel: K(x, x') = sum_i K_i(x_i, x'_i).
    """
    
    is_stationary = True
    has_lengthscale = False

    def __init__(
        self,
        num_dims: int,          
        num_mixtures_per_dim: int = 1, 
        batch_shape: Optional[torch.Size] = torch.Size([]),
        **kwargs,
    ):
        super().__init__(batch_shape=batch_shape, **kwargs)
        
        self.num_dims = num_dims
        
        self.kernels = torch.nn.ModuleList([
            LearnableAlphaStableKernel(
                num_mixtures=num_mixtures_per_dim,
                num_dims=1,                        
                active_dims=[i],                  
                batch_shape=batch_shape,
                **kwargs 
            )
            for i in range(num_dims)
        ])

    def forward(self, x1, x2, diag=False, **params):
        res = 0.0
        for kernel in self.kernels:
            res = res + kernel(x1, x2, diag=diag, **params)
        return res
    
    def initialize_from_data_empspect(self, train_x, train_y):
        """
        Initialize each additive component independently from data.
        """
        print(f"Initializing {self.num_dims} additive components independently...")
        for i, kernel in enumerate(self.kernels):
            try:
                kernel.initialize_from_data_empspect(train_x, train_y)
            except Exception as e:
                print(f"  > Warning: Init failed for dim {i}: {e}")
