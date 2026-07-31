from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData, concat

from sckitflow.data._mixins import MappedArray


def map_int_to_str(values: np.ndarray, descr: str) -> np.ndarray:
    """Prefix every integer in `values` with `descr`, returning a string array.

    The inputs are low-cardinality label codes, so only the distinct values are
    formatted and the result is gathered by index -- `np.vectorize` would run the
    Python-level f-string once per element instead.
    """
    values = np.asarray(values)
    uniques, inverse = np.unique(values, return_inverse=True)
    labels = np.array([f"{descr}{val}" for val in uniques])
    return labels[inverse].reshape(values.shape)


n_obs_pert = 50_000
n_obs_ctrl = 10_000
n_genes = 400
nunique_drugs = 3
nunique_kos = 5
nunique_targets = 5
nunique_source_splits = 5
paired_condition_n_feats = 100
drug_rep_n_feats = 512
ko_rep_n_feats = 512
source_rep_n_feats = 512
n_feats_obsm_repr = 200
continuous_target_dim = 10
src_dim = 16
tgt_dim = 22

uns_keys = ["drug", "ko"]
control_key = "is_control"
is_control_val = "control"
repr_obsm_key = "X_repr"
continuous_target_obsm_key = "target_variable"
src_obsm_key = "X_src"
tgt_obsm_key = "X_tgt"

obs_columns_to_nunique_and_prefix = {
    "drugA": (nunique_drugs, "drug"),
    "drugB": (nunique_drugs, "drug"),
    "koA": (nunique_kos, "ko"),
    "koB": (nunique_kos, "ko"),
    "target": (nunique_targets, "target"),
    "source_split": (nunique_source_splits, "source_split"),
}


obs_columns_to_fixed_val = {
    "drugA": is_control_val,
    "drugB": is_control_val,
    "koA": is_control_val,
    "koB": is_control_val,
}


obsm_keys_to_dim = {
    "drugA_time": 1,
    "drugA_dose": 1,
    "drugB_time": 1,
    "drugB_dose": 1,
    "koA_time": 1,
    "koA_dose": 1,
    "koB_time": 1,
    "koB_dose": 1,
    "paired_condition": paired_condition_n_feats,
    repr_obsm_key: n_feats_obsm_repr,
    continuous_target_obsm_key: continuous_target_dim,
    src_obsm_key: src_dim,
    tgt_obsm_key: tgt_dim,
}

uns_keys_to_nunique_prefix_and_dim = {
    "drug": (nunique_drugs, "drug", drug_rep_n_feats),
    "ko": (nunique_kos, "ko", ko_rep_n_feats),
    "source_split": (nunique_source_splits, "source_split", source_rep_n_feats),
}


def _init_obs(
    n_obs: int,
    obs_columns_to_nunique_and_prefix: dict[str, tuple[int, str]],
    obs_columns_to_fixed_val: dict[str, str] = None,
) -> pd.DataFrame:
    """"""  # noqa
    data_dict = {
        k: map_int_to_str(np.random.choice(v[0], n_obs), v[1]) for k, v in obs_columns_to_nunique_and_prefix.items()
    }
    obs_df = pd.DataFrame(data_dict)
    if obs_columns_to_fixed_val is None:
        obs_columns_to_fixed_val = {}
    for k, v in obs_columns_to_fixed_val.items():
        obs_df[k] = v
    return obs_df


def _init_obsm(n_obs: int, obsm_keys_to_dim: dict[str, int], zeros: bool = False) -> MappedArray:
    """"""  # noqa
    init_fn = lambda n, v: np.zeros((n, v)) if zeros else np.random.rand(n, v)
    return {k: init_fn(n_obs, v) for k, v in obsm_keys_to_dim.items()}


def _init_uns(
    uns_keys_to_nunique_prefix_and_dim: dict[str, Any],
    is_control_val: str,
) -> dict[str, Any]:
    """"""  # noqa

    uns_dict = {}
    for k, v in uns_keys_to_nunique_prefix_and_dim.items():
        uns = {uval.item(): np.random.randn(1, v[2]) for uval in map_int_to_str(np.arange(v[0]), v[1])}
        uns[is_control_val] = np.zeros((1, v[2]))
        uns_dict[k] = uns
    return uns_dict


def _get_perturbed_adata(
    n_obs: int = n_obs_pert,
    n_genes: int = n_genes,
    obsm_keys_to_dim: dict[str, int] = obsm_keys_to_dim,
    obs_columns_to_nunique_and_prefix: Sequence[str] = obs_columns_to_nunique_and_prefix,
    uns_keys_to_nunique_prefix_and_dim: dict[str, Any] = uns_keys_to_nunique_prefix_and_dim,
    is_control_val: str = is_control_val,
) -> AnnData:
    """"""  # noqa

    # initializing X
    X = np.random.randn(n_obs, n_genes)

    # initializing obs
    obs = _init_obs(n_obs, obs_columns_to_nunique_and_prefix)

    # initializing obsm
    obsm = _init_obsm(n_obs, obsm_keys_to_dim, zeros=True)

    # initializing uns
    uns = _init_uns(uns_keys_to_nunique_prefix_and_dim, is_control_val)

    return AnnData(
        X=X,
        obs=obs,
        obsm=obsm,
        uns=uns,
    )


def _get_control_adata(
    n_obs: int = n_obs_ctrl,
    n_genes: int = n_genes,
    obsm_keys_to_dim: dict[str, int] = obsm_keys_to_dim,
    obs_columns_to_nunique_and_prefix: dict[str, tuple[int | str]] = obs_columns_to_nunique_and_prefix,
    uns_keys_to_nunique_prefix_and_dim: dict[str, Any] = uns_keys_to_nunique_prefix_and_dim,
    obs_columns_to_fixed_val: dict[str, str] = obs_columns_to_fixed_val,
    is_control_val: str = is_control_val,
) -> AnnData:
    """"""  # noqa

    # initializing X
    X = np.random.randn(n_obs, n_genes)

    # initializing obs
    obs = _init_obs(n_obs, obs_columns_to_nunique_and_prefix, obs_columns_to_fixed_val)

    # initializing obsm
    obsm = _init_obsm(n_obs, obsm_keys_to_dim, zeros=True)

    # initializing uns
    uns = _init_uns(uns_keys_to_nunique_prefix_and_dim, is_control_val)

    return AnnData(
        X=X,
        obs=obs,
        obsm=obsm,
        uns=uns,
    )


def get_dummy_adata(
    n_obs_pert: int = n_obs_pert,
    n_obs_ctrl: int = n_obs_ctrl,
    n_genes: int = n_genes,
    obsm_keys_to_dim: dict[str, int] = obsm_keys_to_dim,
    obs_columns_to_nunique_and_prefix: Sequence[str] = obs_columns_to_nunique_and_prefix,
    uns_keys_to_nunique_prefix_and_dim: dict[str, Any] = uns_keys_to_nunique_prefix_and_dim,
    obs_columns_to_fixed_val: dict[str, str] = obs_columns_to_fixed_val,
    control_key: str = control_key,
) -> AnnData:
    """"""  # noqa

    # perturbation adata
    pert_adata = _get_perturbed_adata(
        n_obs=n_obs_pert,
        n_genes=n_genes,
        obsm_keys_to_dim=obsm_keys_to_dim,
        obs_columns_to_nunique_and_prefix=obs_columns_to_nunique_and_prefix,
        uns_keys_to_nunique_prefix_and_dim=uns_keys_to_nunique_prefix_and_dim,
    )
    pert_adata.obs[control_key] = False
    pert_adata.obs_names = np.arange(n_obs_pert).astype(str)

    # control adata
    ctrl_adata = _get_control_adata(
        n_obs=n_obs_ctrl,
        n_genes=n_genes,
        obsm_keys_to_dim=obsm_keys_to_dim,
        obs_columns_to_nunique_and_prefix=obs_columns_to_nunique_and_prefix,
        uns_keys_to_nunique_prefix_and_dim=uns_keys_to_nunique_prefix_and_dim,
        obs_columns_to_fixed_val=obs_columns_to_fixed_val,
    )
    ctrl_adata.obs[control_key] = True
    ctrl_adata.obs_names = (n_obs_pert + np.arange(n_obs_ctrl)).astype(str)

    # contantenating the two adatas
    adata = concat((pert_adata, ctrl_adata))
    adata.uns = ctrl_adata.uns
    return adata
