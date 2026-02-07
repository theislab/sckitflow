from collections.abc import Collection, Sequence

import pandas as pd
import pytest

from sc_flow.data.sim._dummy_adata import get_dummy_adata


@pytest.fixture
def n_obs_pert() -> int:
    return 50_000


@pytest.fixture
def n_obs_ctrl() -> int:
    return 10_000


@pytest.fixture
def n_obs(
    n_obs_pert: int,
    n_obs_ctrl: int,
) -> int:
    return n_obs_ctrl + n_obs_pert


@pytest.fixture
def n_genes() -> int:
    return 400


@pytest.fixture
def nunique_drugs() -> int:
    return 3


@pytest.fixture
def nunique_kos() -> int:
    return 5


@pytest.fixture
def nunique_targets() -> int:
    return 5


@pytest.fixture
def nunique_source_splits() -> int:
    return 5


@pytest.fixture
def paired_condition_n_feats() -> int:
    return 100


@pytest.fixture
def drug_rep_n_feats() -> int:
    return 512


@pytest.fixture
def ko_rep_n_feats() -> int:
    return 512


@pytest.fixture
def source_rep_n_feats() -> int:
    return 512


@pytest.fixture
def n_feats_obsm_repr() -> int:
    return 200


@pytest.fixture
def continuous_target_dim() -> int:
    return 10


@pytest.fixture
def src_coupling_dims() -> int:
    return 16


@pytest.fixture
def tgt_coupling_dims() -> int:
    return 20


@pytest.fixture
def uns_keys() -> Sequence[str]:
    return ("drug", "ko", "source_split")


@pytest.fixture
def control_key() -> str:
    return "is_control"


@pytest.fixture
def is_control_val() -> str:
    return "control"


@pytest.fixture
def repr_obsm_key() -> str:
    return "X_repr"


@pytest.fixture
def continuous_target_obsm_key() -> str:
    return "target_variable"


@pytest.fixture
def src_coupling_obsm_key() -> str:
    return "X_src"


@pytest.fixture
def tgt_coupling_obsm_key() -> str:
    return "X_tgt"


@pytest.fixture
def obs_columns_to_nunique_and_prefix(
    nunique_drugs: int,
    nunique_kos: int,
    nunique_targets: int,
    nunique_source_splits: int,
) -> dict[str, tuple[int | str]]:
    return {
        "drugA": (nunique_drugs, "drug"),
        "drugB": (nunique_drugs, "drug"),
        "koA": (nunique_kos, "ko"),
        "koB": (nunique_kos, "ko"),
        "target": (nunique_targets, "target"),
        "source_split": (nunique_source_splits, "source_split"),
    }


@pytest.fixture
def obs_columns_to_fixed_val(
    is_control_val: str,
) -> dict[str, str]:
    return {
        "drugA": is_control_val,
        "drugB": is_control_val,
        "koA": is_control_val,
        "koB": is_control_val,
    }


@pytest.fixture
def obsm_keys_to_dim(
    paired_condition_n_feats: int,
    n_feats_obsm_repr: int,
    continuous_target_dim: int,
    repr_obsm_key: str,
    continuous_target_obsm_key: str,
    src_coupling_dims: int,
    tgt_coupling_dims: int,
    src_coupling_obsm_key: str,
    tgt_coupling_obsm_key: str,
) -> dict[str, int]:
    return {
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
        tgt_coupling_obsm_key: tgt_coupling_dims,
        src_coupling_obsm_key: src_coupling_dims,
    }


@pytest.fixture
def uns_keys_to_nunique_prefix_and_dim(
    nunique_drugs: int,
    nunique_kos: int,
    nunique_source_splits: int,
    drug_rep_n_feats: int,
    ko_rep_n_feats: int,
    source_rep_n_feats: int,
) -> dict[str, tuple[int, str, int]]:
    return {
        "drug": (nunique_drugs, "drug", drug_rep_n_feats),
        "ko": (nunique_kos, "ko", ko_rep_n_feats),
        "source_split": (nunique_source_splits, "source_split", source_rep_n_feats),
    }


@pytest.fixture
def adata(
    n_obs_pert: int,
    n_obs_ctrl: int,
    n_genes: int,
    obsm_keys_to_dim: dict[str, int],
    obs_columns_to_nunique_and_prefix: dict[str, tuple[int | str]],
    uns_keys_to_nunique_prefix_and_dim: dict[str, int | str],
    obs_columns_to_fixed_val: dict[str, str],
    control_key: str,
):
    return get_dummy_adata(
        n_obs_pert=n_obs_pert,
        n_obs_ctrl=n_obs_ctrl,
        n_genes=n_genes,
        obsm_keys_to_dim=obsm_keys_to_dim,
        obs_columns_to_nunique_and_prefix=obs_columns_to_nunique_and_prefix,
        uns_keys_to_nunique_prefix_and_dim=uns_keys_to_nunique_prefix_and_dim,
        obs_columns_to_fixed_val=obs_columns_to_fixed_val,
        control_key=control_key,
    )


@pytest.fixture
def registry() -> dict[str, Collection[str]]:
    return {"level0": ["level0-A", "level0-B"], "level1": ["level1-A", "level1-B"]}


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "cat1": ["a", "b", "a", "c"],
            "cat2": [0, 1, 0, 1],
            "num": [0.1, 0.2, 0.3, 0.4],
        }
    )
