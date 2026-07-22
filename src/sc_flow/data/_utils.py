from collections.abc import Callable, Collection, Mapping

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder

from scfit._types import TargetCovariatesEncoderCls, TargetCovariatesEncodingId

__all__ = [
    "get_covariates_encoders_from_dict",
    "get_covariate_encoder",
    "get_label_encoder",
    "get_one_hot_encoder",
    "convert_to_categorical_in_place",
]


def get_covariates_encoders_from_dict(
    categorical_covs_dict: Mapping[str, TargetCovariatesEncodingId],
    covariates_df: pd.DataFrame,
    fn_dict: Mapping[str, Callable] | None = None,
    inverse_fn_dict: Mapping[str, Callable] | None = None,
) -> Mapping[str, TargetCovariatesEncoderCls]:
    encoder_dict = {}
    fn_dict = {} if fn_dict is None else fn_dict
    inverse_fn_dict = {} if inverse_fn_dict is None else inverse_fn_dict
    for cov_name, enc_id in categorical_covs_dict.items():
        cov_data = covariates_df.loc[:, cov_name].values
        fn = fn_dict.get(cov_name, None)
        inverse_fn = inverse_fn_dict.get(cov_name, None)
        encoder_dict[cov_name] = get_covariate_encoder(enc_id, cov_data, fn=fn, inverse_fn=inverse_fn)
    return encoder_dict


def get_covariate_encoder(
    encoder_id: TargetCovariatesEncodingId,
    data: np.ndarray,
    fn: Callable | None = None,
    inverse_fn: Callable | None = None,
) -> LabelEncoder | OneHotEncoder:
    if encoder_id == "label":
        return get_label_encoder(data)
    elif encoder_id == "one-hot":
        return get_one_hot_encoder(data)
    elif encoder_id == "functional":
        # return FunctionTransformer(func=fn, inverse_func=inverse_fn, check_inverse=False)
        return get_functional_encoder(data=data, func=fn, inverse_func=inverse_fn)
    msg = f"Covariate Encoder {encoder_id} not available.Possible options are {TargetCovariatesEncodingId}"
    raise ValueError(msg)


def get_functional_encoder(data: np.ndarray, func: Callable, inverse_func: Callable) -> FunctionTransformer:
    return FunctionTransformer(func=func, inverse_func=inverse_func, check_inverse=False).fit(data)


def get_label_encoder(data: np.ndarray) -> LabelEncoder:
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
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    if data.ndim != 2:
        msg = (
            'When using "one-hot" as target representation in `target_covariates`,'
            f"you need to pass a 2-dimensional array, found {data.ndim=}."
        )
        raise ValueError(msg)
    return OneHotEncoder().fit(data)


def convert_to_categorical_in_place(df: pd.DataFrame, cols: Collection[str] | None) -> None:
    if cols is None:
        return
    for c in cols:
        if c in df.columns and not hasattr(df[c], "cat"):
            df[c] = df[c].astype("category")
