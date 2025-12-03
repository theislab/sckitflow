from collections.abc import Collection

from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder

from sc_flow._types import TargetCovariatesEncodingId
from sc_flow.data._mixins import BatchMixin
from sc_flow.data._structures import CategoricalData

__all__ = [
    "verify_repr_shape",
    "verify_repr",
    "verify_categorical_data",
    "verify_mixin",
]


def verify_repr_shape(
    data: CategoricalData,
    condition: str,
    uns_keys_to_nunique_prefix_and_dim: dict[str, tuple[int, str, int]],
) -> None:
    cond_repr_dict = data.repr_dict[condition]
    for v in cond_repr_dict.values():
        emb_dim = uns_keys_to_nunique_prefix_and_dim[condition][-1]
        expected_shape = (1, emb_dim)
        assert v.shape == expected_shape


def verify_repr(
    data: CategoricalData,
    uns_keys_to_nunique_prefix_and_dim: dict[str, tuple[int, str, int]],
    conditions_reps: dict[str, str] | None = None,
    groups_encoding: dict[str, TargetCovariatesEncodingId] | None = None,
) -> None:
    if conditions_reps is not None:
        for condition, value in conditions_reps.items():
            assert value in data.repr_dict
            verify_repr_shape(data, condition, uns_keys_to_nunique_prefix_and_dim)
    if groups_encoding is not None:
        for cov, encoder_id in groups_encoding.items():
            assert cov in data.categorical_encoders
            encoder = data.categorical_encoders[cov]
            if encoder_id == "one-hot":
                assert isinstance(encoder, OneHotEncoder)
            if encoder_id == "label":
                assert isinstance(encoder, LabelEncoder)
            if encoder_id == "identity":
                assert isinstance(encoder, FunctionTransformer)


def verify_categorical_data(
    data: CategoricalData,
    expected_df_cols: Collection[str],
    uns_keys_to_nunique_prefix_and_dim: dict[str, tuple[int, str, int]],
    conditions_reps: dict[str, str] | None = None,
    groups_encoding: dict[str, TargetCovariatesEncodingId] | None = None,
) -> None:
    assert set(data.ann_df.columns) == set(expected_df_cols)
    verify_repr(
        data, uns_keys_to_nunique_prefix_and_dim, conditions_reps=conditions_reps, groups_encoding=groups_encoding
    )


def verify_mixin(
    data: BatchMixin,
    N: int,
    obsm_keys_to_dim: dict[str, int],
    covariates: Collection[str] | None,
):
    if covariates is None:
        return None
    for condition in covariates:
        assert condition in data.mapping
        val = data.mapping[condition]
        emb_dim = obsm_keys_to_dim[condition]
        expected_shape = (N, emb_dim)
        assert val.shape == expected_shape
