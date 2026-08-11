import copy
from collections.abc import Callable, Collection, Mapping
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder

from sckitflow._types import TargetCovariatesEncoderCls, TargetCovariatesEncodingId

__all__ = [
    "get_covariates_encoders_from_dict",
    "get_covariate_encoder",
    "get_label_encoder",
    "get_one_hot_encoder",
    "convert_to_categorical_in_place",
    "with_derived_obs",
]


def with_derived_obs(adata: AnnData, **cols: Any) -> AnnData:
    """A shallow AnnData sharing ``adata``'s arrays but carrying extra ``.obs`` columns.

    For a column sckitflow derives -- a split label, the implicit all-cells group, the pair id behind fixed
    matching. Such a column has to be in the obs the consumer reads (scfit reads grouping only from ``.obs``),
    but it has no business being in the caller's object: writing one there mutates their AnnData, and on a
    **view** it materializes the whole thing (anndata's "initializing view as actual") -- the copy an
    out-of-core path exists to avoid. ``adata.copy()`` is worse still: a deep copy of ``X`` to add one column.

    ``copy.copy`` shares ``X`` / ``obsm`` / ``uns`` by reference, and the obs frame is copied *shallowly*:
    a new frame whose existing columns still point at the original buffers, with the derived ones added to
    it. Nothing is duplicated but the frame itself -- adding a column to a shallow copy cannot write through
    to the original, while ``assign`` would deep-copy the columns on a pandas without copy-on-write.

    The result is therefore **read-only by convention** -- add derived columns, read everything else. Nothing
    enforces it: ``X`` and ``obsm`` are the caller's own arrays, so a write to them lands in ``adata``, and
    on a pandas without copy-on-write (off by default before 3.0) so can a write to an *existing* obs column.
    That cannot be prevented without copying the cell data this helper exists to share -- marking the arrays
    non-writeable would disable writes on the caller's AnnData too, since it is the same array object. (A view
    still materializes its own subset -- a view's obs is a slice -- but the caller's object stays untouched.)
    """
    local = copy.copy(adata)
    obs = adata.obs.copy(deep=False)
    for name, values in cols.items():
        obs[name] = values
    local.obs = obs
    return local


def get_covariates_encoders_from_dict(
    categorical_covs_dict: Mapping[str, TargetCovariatesEncodingId],
    covariates_df: pd.DataFrame,
    fn_dict: Mapping[str, Callable] | None = None,
    inverse_fn_dict: Mapping[str, Callable] | None = None,
) -> Mapping[str, TargetCovariatesEncoderCls]:
    """Configures the covariates encoder from a dictionary of transforms per column and input data.

    :param categorical_covs_dict: Dictionary mapping each categorical column to encode to the
        corresponding identifier of the tranform to apply. Note that each key of this dictionary
        should appear as column in :param: `covariate_df`.
    :type categorical_covs_dict: class: `Mapping[str, TargetCovariatesEncodingId]`

    :param covariates_df: The dataframe containing the data used to fit the tranformations
    :type covariates_df: class:`pd.DataFrame`

    :param fn_dict: Optional dictionary used to define the transormation for the functional tranform.
        Its keys should appear in :param: `categorical_covs_dict` too.
        Will only be considered for those keys whose corresponding value in
        :param: `categorical_covs_dict` is `"functional"`, otherwise such keys are ignored.
    :type fn_dict: class:`Mapping[str, Callable] | None = None`

    :param inverse_fn_dict: Optional dictionary used to define the inverse transormation
        for the functional tranform. Its keys should appear in :param: `categorical_covs_dict` too.
        Will only be considered for those keys whose corresponding value in
        :param: `categorical_covs_dict` is `"functional"`, otherwise such keys are ignored.
    :type inverse_fn_dict: class:`Mapping[str, Callable] | None = None`
    """
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
    """Fits the provided encoders to the input data.

    :param encoder_id: The string identifier for the encoder to be used.
    :type encoder_id: class: `TargetCovariatesEncodingId`

    :param data: The input data to fit the encoder on.
    :type data: class: `np.ndarray`

    :param fn: Optional function used to define the transormation for the functional tranform.
        Will only be used when :param: `encoder_id` is `"functional"`, ignored otherwise.
        Defaults to `None`, in which case it will fall back to identity function when used.
    :type fn: class: `Callable | None`

    :param inverse_fn: Optional function used to define the inverse transormation for the functional tranform.
        Will only be used when :param: `encoder_id` is `"functional"`, ignored otherwise.
        Defaults to `None`, in which case it will fall back to identity function when used.
    :type inverse_fn: class: `Callable | None`
    """
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
    """Fit a custom function on the provided data

    :param data: Array contraining the input data to fit the custom function on
        The input dimensions is whatever the custom function expects.
    :type data: class: `np.ndarray`
    """
    return FunctionTransformer(func=func, inverse_func=inverse_func, check_inverse=False).fit(data)


def get_label_encoder(data: np.ndarray) -> LabelEncoder:
    """Fits a label encoder on the provided data.

    :param data: Array containing the input data to fit the label encoder on.
        It should be either a one dimensional array or a two dimensional array
        with singleton trailing dim, in which case the trailing dimension will be removed.
        This is needed for compatibility with :class: `sklearn.preprocessing.LabelEncoder`.
    :type data: class: `np.ndarray`
    """
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
    """Fits a one hot encoder on the provided data.

    :param data: Array containing the input data to fit the label encoder on.
        It should be either a one dimensional array or a two dimensional array
        with singleton trailing dim. In the former case, a singleton trailing dimension
        will be added. This is needed for compatibility with :class: `sklearn.preprocessing.OneHotEncoder`.
    :type data: class: `np.ndarray`
    """
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
    """Convert the given columns to categorical in place."""
    if cols is None:
        return
    for c in cols:
        if c in df.columns and not hasattr(df[c], "cat"):
            df[c] = df[c].astype("category")
