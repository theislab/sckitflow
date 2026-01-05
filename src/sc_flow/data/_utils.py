from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder

from sc_flow._jit import maybe_jit_fn
from sc_flow._types import TargetCovariatesEncoderCls, TargetCovariatesEncodingId

__all__ = [
    "get_covariates_encoders_from_dict",
    "get_covariate_encoder",
    "get_label_encoder",
    "get_one_hot_encoder",
    "sample_indices_uniformly",
]


def get_covariates_encoders_from_dict(
    categorical_covs_dict: Mapping[str, TargetCovariatesEncodingId],
    covariates_df: pd.DataFrame,
) -> Mapping[str, TargetCovariatesEncoderCls]:
    """"""  # noqa
    encoder_dict = {}
    for cov_name, enc_id in categorical_covs_dict.items():
        cov_data = covariates_df.loc[:, cov_name].values
        encoder_dict[cov_name] = get_covariate_encoder(enc_id, cov_data)
    return encoder_dict


def get_covariate_encoder(
    encoder_id: TargetCovariatesEncodingId,
    data: np.ndarray,
) -> LabelEncoder | OneHotEncoder:
    """"""  # noqa
    if encoder_id == "label":
        return get_label_encoder(data)
    elif encoder_id == "one-hot":
        return get_one_hot_encoder(data)
    elif encoder_id == "identity":
        return FunctionTransformer()
    msg = f"Covariate Encoder {encoder_id} not available.Possible options are {TargetCovariatesEncodingId}"
    raise ValueError(msg)


def get_label_encoder(data: np.ndarray) -> LabelEncoder:
    """Fits a label encoder on the provided data."""
    if data.ndim == 2:
        if data.shape[1] != 1:
            msg = (
                'When using "label"  as target representation in `target_covariates`,'
                f"the last dimension should be singleton, but found data of shape {data.shape}."
            )
            raise ValueError(msg)
        data = data.reshape(-1)

    # we require only one dimension when using label encoding
    if data.ndim != 1:
        msg = (
            'When using "label" as target representation in `target_covariates`,'
            f"the data should have only one dimension, but found data of shape {data.shape}."
        )
        raise ValueError(msg)
    return LabelEncoder().fit(data)


def get_one_hot_encoder(data: np.ndarray) -> OneHotEncoder:
    """Fits a one hot encoder on the provided data."""
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    if data.ndim != 2:
        msg = (
            'When using "one-hot" as target representation in `target_covariates`,'
            f"you need to pass a 2-dimensional array, found {data.ndim=}."
        )
        raise ValueError(msg)
    return OneHotEncoder().fit(data)


@maybe_jit_fn(jit_kwargs={"nopython": True})
def sample_indices_uniformly(n_obs: int, batch_size: int, replace: bool) -> np.ndarray:
    if replace:
        return np.random.randint(0, n_obs, batch_size)
    else:
        return np.random.permutation(n_obs)[:batch_size]
