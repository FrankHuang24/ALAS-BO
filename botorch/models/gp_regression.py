#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Gaussian Process Regression models based on GPyTorch models.

This file is a lightly modified version to support custom kernel strings:
- "alas"      : coupled ALAS (LearnableAlphaStableKernel)
- "alas_sep"  : separable/additive ALAS (AdditiveLearnableAlphaStableKernel)

Backward-compatible aliases are also supported:
- "learnable_alpha_stable" -> "alas"
- "additive_stable"        -> "alas_sep"
- "ma52"                   -> Matérn-5/2 baseline (alias)
"""

from __future__ import annotations

import warnings
from typing import Any, List, Optional, Union

import torch
from torch import Tensor

from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.model import FantasizeMixin
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform, Standardize
from botorch.models.utils import validate_input_scaling
from botorch.models.utils.gpytorch_modules import (
    get_covar_module_with_dim_scaled_prior,
    get_gaussian_likelihood_with_lognormal_prior,
)
from botorch.utils.containers import BotorchContainer
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.types import _DefaultType, DEFAULT

from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels import (
    Kernel,
    LinearKernel,
    MaternKernel,
    PeriodicKernel,
    RBFKernel,
    RQKernel,
    ScaleKernel,
    SpectralDeltaKernel,
    SpectralMixtureKernel,
)
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.means.mean import Mean
from gpytorch.models.exact_gp import ExactGP
from gpytorch.module import Module


# Optional imports (kept silent for clean release)
MixedFixedAlphaStableKernel = None
LearnableAlphaStableKernel = None
AdditiveLearnableAlphaStableKernel = None
CauchyMixtureKernel = None
AdaptiveKernel = None

try:
    from botorch.models.kernels.stable_kernel import (
        LearnableAlphaStableKernel,
        AdditiveLearnableAlphaStableKernel,
    )
except Exception:
    pass

try:
    from botorch.models.kernels.adaptive_kernel import AdaptiveKernel
except Exception:
    pass

try:
    from botorch.models.kernels.cauchy_spectral_mixture import CauchyMixtureKernel
except Exception:
    pass


class SingleTaskGP(BatchedMultiOutputGPyTorchModel, ExactGP, FantasizeMixin):
    r"""A single-task exact GP model, supporting both known and inferred noise levels."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        likelihood: Likelihood | None = None,
        covar_module: Union[Module, str, None] = None,
        n_mixture: Optional[int] = None,
        mean_module: Mean | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
        n_mixture1: Optional[int] = None,
        n_mixture2: Optional[int] = None,
        alphas: Optional[List[float]] = None,
    ) -> None:
        self._original_train_X = train_X
        self._original_train_Y = train_Y

        self._validate_tensor_args(X=train_X, Y=train_Y, Yvar=train_Yvar)

        if outcome_transform == DEFAULT:
            outcome_transform = Standardize(
                m=train_Y.shape[-1], batch_shape=train_X.shape[:-2]
            )

        # Apply input transform (if any) before scaling checks / dimension setting.
        self._input_transform_applied = False
        if input_transform is not None:
            if not isinstance(input_transform, Module):
                input_transform = torch.nn.ModuleList([input_transform])
            self.input_transform = input_transform
            transformed_X = self.transform_inputs(X=train_X)
            self._input_transform_applied = True
        else:
            transformed_X = train_X

        # Apply outcome transform (if any).
        if outcome_transform is not None:
            train_Y_t, train_Yvar_t = outcome_transform(train_Y, train_Yvar)
        else:
            train_Y_t, train_Yvar_t = train_Y, train_Yvar

        self._validate_tensor_args(X=transformed_X, Y=train_Y_t, Yvar=train_Yvar_t)

        ignore_X_dims = getattr(self, "_ignore_X_dims_scaling_check", None)
        validate_input_scaling(
            train_X=transformed_X,
            train_Y=train_Y_t,
            train_Yvar=train_Yvar_t,
            ignore_X_dims=ignore_X_dims,
        )

        # Dimensions are defined using the *original* inputs.
        self._set_dimensions(train_X=train_X, train_Y=train_Y)

        # Transform args for GPyTorch init (uses transformed tensors).
        t_train_X, t_train_Y, t_train_Yvar = self._transform_tensor_args(
            X=transformed_X, Y=train_Y_t, Yvar=train_Yvar_t
        )

        # Likelihood
        if likelihood is None:
            if train_Yvar is None:
                likelihood = get_gaussian_likelihood_with_lognormal_prior(
                    batch_shape=self._aug_batch_shape
                )
            else:
                likelihood = FixedNoiseGaussianLikelihood(
                    noise=t_train_Yvar, batch_shape=self._aug_batch_shape
                )
        else:
            self._is_custom_likelihood = True

        # Kernel handling
        ard_num_dims = transformed_X.shape[-1]
        batch_shape = self._aug_batch_shape

        covar: Module
        if isinstance(covar_module, Module):
            covar = covar_module
        elif isinstance(covar_module, str) or covar_module is None:
            kname = "default" if covar_module is None else covar_module

            # Backward-compatible aliases
            if kname == "learnable_alpha_stable":
                kname = "alas"
            if kname == "additive_stable":
                kname = "alas_sep"
            if kname == "ma52":
                kname = "matern52"

            if kname == "rbf":
                base = RBFKernel(ard_num_dims=ard_num_dims, batch_shape=batch_shape)
                covar = ScaleKernel(base, batch_shape=batch_shape)

            elif kname in ("matern", "matern52"):
                base = MaternKernel(nu=2.5, ard_num_dims=ard_num_dims, batch_shape=batch_shape)
                covar = ScaleKernel(base, batch_shape=batch_shape)

            elif kname == "matern32":
                base = MaternKernel(nu=1.5, ard_num_dims=ard_num_dims, batch_shape=batch_shape)
                covar = ScaleKernel(base, batch_shape=batch_shape)

            elif kname == "matern12":
                base = MaternKernel(nu=0.5, ard_num_dims=ard_num_dims, batch_shape=batch_shape)
                covar = ScaleKernel(base, batch_shape=batch_shape)

            elif kname == "rq":
                base = RQKernel(ard_num_dims=ard_num_dims, batch_shape=batch_shape)
                covar = ScaleKernel(base, batch_shape=batch_shape)

            elif kname == "pe":
                base = PeriodicKernel(ard_num_dims=ard_num_dims, batch_shape=batch_shape)
                covar = ScaleKernel(base, batch_shape=batch_shape)

            elif kname == "gsm":
                if n_mixture is None:
                    raise ValueError("n_mixture is required for 'gsm' kernel.")
                base = SpectralMixtureKernel(
                    num_mixtures=n_mixture,
                    ard_num_dims=ard_num_dims,
                    batch_shape=batch_shape,
                )
                covar = ScaleKernel(base, batch_shape=batch_shape)

            elif kname == "csm":
                if CauchyMixtureKernel is None:
                    raise ImportError("CauchyMixtureKernel is not available.")
                if n_mixture is None:
                    raise ValueError("n_mixture is required for 'csm' kernel.")
                base = CauchyMixtureKernel(
                    num_mixtures=n_mixture,
                    ard_num_dims=ard_num_dims,
                    batch_shape=batch_shape,
                )
                covar = ScaleKernel(base, batch_shape=batch_shape)

            elif kname == "mix":
                if CauchyMixtureKernel is None:
                    raise ImportError("CauchyMixtureKernel is not available.")
                if n_mixture1 is None or n_mixture2 is None:
                    raise ValueError("n_mixture1 and n_mixture2 are required for 'mix' kernel.")
                k1 = CauchyMixtureKernel(
                    num_mixtures=n_mixture1, ard_num_dims=ard_num_dims, batch_shape=batch_shape
                )
                k2 = SpectralMixtureKernel(
                    num_mixtures=n_mixture2, ard_num_dims=ard_num_dims, batch_shape=batch_shape
                )
                covar = ScaleKernel(k1 + k2, batch_shape=batch_shape)

            elif kname == "sdk":
                base = SpectralDeltaKernel(num_dims=ard_num_dims, batch_shape=batch_shape)
                covar = ScaleKernel(base, batch_shape=batch_shape)

            elif kname == "ada":
                if AdaptiveKernel is None:
                    raise ImportError("AdaptiveKernel is not available.")
                kernel_list = [
                    RBFKernel(batch_shape=batch_shape),
                    MaternKernel(nu=2.5, batch_shape=batch_shape),
                    LinearKernel(batch_shape=batch_shape),
                    RQKernel(batch_shape=batch_shape),
                ]
                base = AdaptiveKernel(kernel_list, batch_shape=batch_shape)
                covar = ScaleKernel(base, batch_shape=batch_shape)

            elif kname == "mixstable_fixed_alpha":
                if MixedFixedAlphaStableKernel is None:
                    raise ImportError("MixedFixedAlphaStableKernel is not available.")
                if alphas is None or len(alphas) == 0:
                    raise ValueError("`alphas` must be provided for 'mixstable_fixed_alpha'.")
                base = MixedFixedAlphaStableKernel(
                    alphas=alphas,
                    num_dims=ard_num_dims,
                    batch_shape=batch_shape,
                )
                covar = ScaleKernel(base, batch_shape=batch_shape)

            elif kname == "alas":
                if LearnableAlphaStableKernel is None:
                    raise ImportError("LearnableAlphaStableKernel is not available.")
                if n_mixture is None:
                    raise ValueError("`n_mixture` (Q) must be provided for 'alas'.")
                base = LearnableAlphaStableKernel(
                    num_mixtures=n_mixture,
                    num_dims=ard_num_dims,
                    batch_shape=batch_shape,
                )
                covar = ScaleKernel(base, batch_shape=batch_shape)

            elif kname == "alas_sep":
                if AdditiveLearnableAlphaStableKernel is None:
                    raise ImportError("AdditiveLearnableAlphaStableKernel is not available.")
                q_per_dim = 1 if n_mixture is None else int(n_mixture)
                base = AdditiveLearnableAlphaStableKernel(
                    num_dims=ard_num_dims,
                    num_mixtures_per_dim=q_per_dim,
                    batch_shape=batch_shape,
                )
                covar = ScaleKernel(base, batch_shape=batch_shape)

            else:
                warnings.warn(
                    f"Unknown kernel string '{kname}'. Using default covar module.",
                    UserWarning,
                )
                covar = get_covar_module_with_dim_scaled_prior(
                    ard_num_dims=ard_num_dims, batch_shape=batch_shape
                )
        else:
            raise TypeError("covar_module must be a Module, a string, or None.")

        # Initialize ExactGP
        ExactGP.__init__(
            self,
            train_inputs=t_train_X,
            train_targets=t_train_Y,
            likelihood=likelihood,
        )

        if mean_module is None:
            mean_module = ConstantMean(batch_shape=batch_shape)
        self.mean_module = mean_module
        self.covar_module = covar

        # Data-driven kernel initialization (heuristic; silent on success/failure).
        initialized = False
        init_X, init_Y = t_train_X, t_train_Y
        if hasattr(self.covar_module, "initialize_from_data_empspect"):
            try:
                self.covar_module.initialize_from_data_empspect(init_X, init_Y)
                initialized = True
            except Exception:
                pass

        if (not initialized) and hasattr(self.covar_module, "initialize_from_data"):
            try:
                self.covar_module.initialize_from_data(init_X, init_Y)
                initialized = True
            except Exception:
                pass

        # Subset batch dict (best-effort; supports botorch batching utilities).
        self._subset_batch_dict = {"mean_module.raw_constant": -1}
        if train_Yvar is None and hasattr(self.likelihood, "noise_covar"):
            self._subset_batch_dict["likelihood.noise_covar.raw_noise"] = -2
        if hasattr(self.covar_module, "raw_outputscale"):
            self._subset_batch_dict["covar_module.raw_outputscale"] = -1
        if hasattr(self.covar_module, "base_kernel") and hasattr(
            self.covar_module.base_kernel, "raw_lengthscale"
        ):
            self._subset_batch_dict["covar_module.base_kernel.raw_lengthscale"] = -3
        elif hasattr(self.covar_module, "raw_lengthscale"):
            self._subset_batch_dict["covar_module.raw_lengthscale"] = -3

        if outcome_transform is not None:
            self.outcome_transform = outcome_transform

        # Move to correct dtype/device.
        self.to(self._original_train_X)

    @classmethod
    def construct_inputs(
        cls, training_data: SupervisedDataset, **kwargs: Any
    ) -> dict[str, BotorchContainer | Tensor]:
        return super().construct_inputs(training_data=training_data, **kwargs)

    def forward(self, x: Tensor) -> MultivariateNormal:
        processed_x = x
        if self.training and hasattr(self, "input_transform"):
            processed_x = self.transform_inputs(x)

        mean_x = self.mean_module(processed_x)
        covar_x = self.covar_module(processed_x)
        return MultivariateNormal(mean_x, covar_x)
