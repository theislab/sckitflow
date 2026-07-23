from __future__ import annotations

from collections.abc import Mapping

import torch

from scfit._utils import check_sequence_query_against_reference
from sc_flow.flow._torch_types import MappedTensor
from sc_flow.flow._config import SetEncoderConfig

__all__ = ["SetEncoder"]


class SetEncoder(torch.nn.Module):
    """Permutation-invariant Deep-Sets encoder over a *set* of perturbation covariates.

    Dumb by construction: :meth:`SetEncoderConfig.build` sizes and assembles the ``ModuleDict`` and hands
    it here. This module only runs ``forward`` and exposes the runtime properties consumed by the velocity
    field.
    """

    def __init__(self, config: SetEncoderConfig, modules: torch.nn.ModuleDict) -> None:
        super().__init__()
        self._config = config
        self._condition_encoder = modules

    @property
    def output_dim(self) -> int:
        return self._config.output_dim

    @property
    def is_stochastic(self) -> bool:
        return self._config.is_stochastic

    @property
    def covariates_pooled(self) -> list[str]:
        return self._config.covariates_pooled

    def forward(
        self,
        condition_dict: MappedTensor,
        condition_mask: Mapping[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not condition_dict:
            raise ValueError("No condition covariate found.")

        # check that the right keys are present
        check_sequence_query_against_reference(
            condition_dict.keys(),
            self._condition_encoder["input_layers"].keys(),
            allow_missing_from_reference=False,
            allow_missing_from_query=False,
        )
        if condition_mask is not None:
            expected_masks = set(self.covariates_pooled)
            supplied_masks = set(condition_mask)
            if supplied_masks != expected_masks:
                missing = sorted(expected_masks - supplied_masks)
                unexpected = sorted(supplied_masks - expected_masks)
                raise ValueError(
                    "An explicit condition_mask must contain every pooled covariate and no others; "
                    f"missing={missing}, unexpected={unexpected}."
                )

        # prepare dictionary to store encoded covariates
        encoded_covariates_to_pool = {}
        encoded_covariates_not_pooled = {}
        pooled_masks = None if condition_mask is None else []
        covariates_not_pooled = self._config.covariates_not_pooled

        # iterating over perturbation covariates
        for covariate_id, covariate_data in condition_dict.items():
            # check that the covariate is present in the data
            if covariate_id not in self._condition_encoder["input_layers"].keys():
                msg = f"Input encoder not found for covariate {covariate_id}"
                raise KeyError(msg)

            # get covariate latent representation
            cov_enc = self._condition_encoder["input_layers"][covariate_id]
            z_cov = cov_enc(covariate_data)

            # update dictionaries
            if covariate_id in covariates_not_pooled:
                if z_cov.ndim != 2:
                    raise ValueError(
                        f"Bypassed covariate {covariate_id!r} must encode to (batch, features), "
                        f"found {tuple(z_cov.shape)}."
                    )
                encoded_covariates_not_pooled[covariate_id] = z_cov

            else:
                # get shared projection layer
                cov_proj = self._condition_encoder["proj_layers"][covariate_id]

                # apply projection and update dict
                z_cov = cov_proj(z_cov)
                if z_cov.ndim != 3:
                    raise ValueError(
                        f"Pooled covariate {covariate_id!r} must encode to (batch, set, features), "
                        f"found {tuple(z_cov.shape)}."
                    )
                encoded_covariates_to_pool[covariate_id] = z_cov
                if pooled_masks is not None:
                    mask = condition_mask[covariate_id]
                    if mask.shape != z_cov.shape[:2]:
                        raise ValueError(
                            f"Mask for {covariate_id!r} must have shape {tuple(z_cov.shape[:2])}, "
                            f"found {tuple(mask.shape)}."
                        )
                    pooled_masks.append(mask)

        # pooled covariates
        if len(encoded_covariates_to_pool) > 0:
            pooled_covariates = torch.concatenate(tuple(encoded_covariates_to_pool.values()), dim=-2)
            combined_mask = None if pooled_masks is None else torch.concatenate(pooled_masks, dim=-1)
            pooled_covariates = self._condition_encoder["pooling_layer"](pooled_covariates, combined_mask)
        else:
            pooled_covariates = None

        # not pooled covariates
        if len(encoded_covariates_not_pooled) > 0:
            covariates_not_pooled_z = torch.concatenate(tuple(encoded_covariates_not_pooled.values()), dim=-1)
        else:
            covariates_not_pooled_z = None

        # get joint representation
        to_concat = []
        if pooled_covariates is not None:
            to_concat.append(pooled_covariates)
        if covariates_not_pooled_z is not None:
            to_concat.append(covariates_not_pooled_z)
        latent_cond = torch.concatenate(to_concat, dim=-1)

        mean = self._condition_encoder["output_layer"](latent_cond)
        logvar = self._condition_encoder["var_layer"](latent_cond) if self.is_stochastic else None
        return mean, logvar
