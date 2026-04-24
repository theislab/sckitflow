from dataclasses import dataclass

import pandas as pd
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder

from sc_flow._types import TargetCovariatesEncoderCls
from sc_flow.data._mixins import BatchMixin
from sc_flow.data.containers._categorical import CategoricalData
from sc_flow.data.containers._coupling import CouplingData
from sc_flow.data.containers._distribution import DistributionData

__all__ = ["DataDimensionalitiesRegistry"]


@dataclass(frozen=True)
class DataDimensionalitiesRegistry:
    state_dim: int
    feature_names: pd.Index
    condition_reps_dims: dict[str, int] | None
    condition_continuous_dims: dict[str, int] | None
    response_reps_dims: dict[str, int] | None
    response_continuous_dims: dict[str, int] | None
    groups_reps_dims: dict[str, int] | None
    source_lin_dim: int | None
    source_quad_dim: int | None
    target_lin_dim: int | None
    target_quad_dim: int | None

    @classmethod
    def _get_dims_from_continuous_data(cls, data: BatchMixin) -> dict[str, int]:
        return {cov_name: cov_data.shape[-1] for cov_name, cov_data in data.mapping.items()}

    @classmethod
    def _get_dims_from_coupling_data(cls, data: CouplingData) -> tuple[int | None, int | None]:
        state_lin_dim = data.state_lin.X.shape[-1] if data.state_lin is not None else None
        state_quad_dim = data.state_quad.X.shape[-1] if data.state_quad is not None else None
        return state_lin_dim, state_quad_dim

    @classmethod
    def _get_functional_tranforms_dim(
        cls,
        cov: str,
        encoder: TargetCovariatesEncoderCls,
        ann_df: pd.DataFrame,
    ) -> int:
        col_vals = ann_df[cov].iloc[:2].values.reshape(-1, 1)
        transformed_vals = encoder.transform(col_vals)
        return transformed_vals.shape[-1]

    @classmethod
    def _get_dims_from_categorical_data(cls, data: CategoricalData) -> dict[str, int]:
        repr_dims = {}
        if data.repr_dict and len(data.repr_dict):
            for cov, cov_repr in data.repr_dict.items():
                if isinstance(cov_repr, dict) and len(cov_repr) > 0:
                    repr_dims[cov] = next(iter(cov_repr.values())).shape[-1]

        encoded_covs_dims = {}
        if data.categorical_encoders and len(data.categorical_encoders):
            for cov, encoder in data.categorical_encoders.items():
                if isinstance(encoder, OneHotEncoder):
                    encoded_covs_dims[cov] = len(encoder.get_feature_names_out())
                elif isinstance(encoder, LabelEncoder):
                    encoded_covs_dims[cov] = 1
                elif isinstance(encoder, FunctionTransformer):
                    # Assumptions: 1-to-1 mapping.
                    # If using polynomial expansion, this will need metadata.
                    encoded_covs_dims[cov] = cls._get_functional_tranforms_dim(cov, encoder, data.ann_df)

        return {**repr_dims, **encoded_covs_dims}

    @classmethod
    def init_from_distribution_data(
        cls, data: DistributionData, feature_names: pd.Index
    ) -> "DataDimensionalitiesRegistry":
        state_dim = data.state_data.X.shape[-1]

        # Helper closures to handle None-checks cleanly
        def get_cat(container):
            if container is not None:
                return (
                    cls._get_dims_from_categorical_data(container.categorical_covariates)
                    if container.categorical_covariates is not None
                    else {}
                )
            return {}

        def get_cont(container):
            if container is not None:
                return (
                    cls._get_dims_from_continuous_data(container.continuous_covariates)
                    if container.continuous_covariates is not None
                    else {}
                )
            return {}

        # Extract dimensions
        condition_reps_dims = get_cat(data.condition_data)
        condition_continuous_dims = get_cont(data.condition_data)

        response_reps_dims = get_cat(data.response_data)
        response_continuous_dims = get_cont(data.response_data)

        # Handle optional groups data
        groups_reps_dims = cls._get_dims_from_categorical_data(data.groups_data) if data.groups_data is not None else {}

        source_lin_dim, source_quad_dim = cls._get_dims_from_coupling_data(data.source_coupling_data)
        target_lin_dim, target_quad_dim = cls._get_dims_from_coupling_data(data.target_coupling_data)

        return cls(
            state_dim=state_dim,
            feature_names=feature_names,
            condition_reps_dims=condition_reps_dims,
            condition_continuous_dims=condition_continuous_dims,
            response_reps_dims=response_reps_dims,
            response_continuous_dims=response_continuous_dims,
            groups_reps_dims=groups_reps_dims,
            source_lin_dim=source_lin_dim,
            source_quad_dim=source_quad_dim,
            target_lin_dim=target_lin_dim,
            target_quad_dim=target_quad_dim,
        )
