"""A dataset split across files keeps each side's reps where its cells are: the loaders route by role."""

import numpy as np
import pytest
from anndata import AnnData

from sckitflow.data import DataManagerConfig, FlowDataModule
from sckitflow.data._group_encoders import LabelEncoderConfig
from sckitflow.data.sim import get_dummy_adata


def _split_files() -> tuple[AnnData, AnnData]:
    """Controls with only the source-side rep, perturbed cells with only the target-side one."""
    adata = get_dummy_adata(
        n_obs_pert=120,
        n_obs_ctrl=40,
        n_genes=10,
        obsm_keys_to_dim={"X_pca": 6},
        obs_columns_to_nunique_and_prefix={"drug": (3, "drug"), "cell_line": (2, "cell_line")},
        uns_keys_to_nunique_prefix_and_dim={"drug": (3, "drug", 4)},
        obs_columns_to_fixed_val={"drug": "control"},
        rng=np.random.default_rng(0),
    )
    adata.obs_names = adata.obs_names.astype(str)
    is_ctrl = (adata.obs["drug"] == "control").to_numpy()
    ctrl, pert = adata[is_ctrl].copy(), adata[~is_ctrl].copy()
    ctrl.obsm["X_src"] = np.full((ctrl.n_obs, 4), 1.0, "float32")
    pert.obsm["X_tgt"] = np.full((pert.n_obs, 3), 2.0, "float32")
    return ctrl, pert


def _manager():
    return DataManagerConfig(
        sample_rep="X_pca",
        categorical_conditions={"drug": ("drug",)},
        categorical_reps={"drug": "drug"},
        control_values_dict={"drug": "control"},
        groups=("cell_line",),
        groups_encoding={"cell_line": LabelEncoderConfig(classes=("cell_line0", "cell_line1"))},
        source_rep_quad="X_src",
        target_rep_quad="X_tgt",
        use_linear=False,
    ).build()


def test_each_side_reads_its_rep_from_its_own_file():
    ctrl, pert = _split_files()
    batch = next(iter(_manager().get_dataloaders([ctrl, pert], batch_size=8)["train"]))
    assert (batch["source_coupling_quad"] == 1.0).all()  # from the controls' file
    assert (batch["target_coupling_quad"] == 2.0).all()  # from the perturbed file


def test_eval_routes_by_role_too():
    ctrl, pert = _split_files()
    step, _leaf = next(iter(_manager().get_eval_loader([ctrl, pert])))
    assert step["source_state"].shape[-1] == 6 and step["target_state"].shape[-1] == 6


def test_missing_rep_names_the_file_role_and_field():
    ctrl, pert = _split_files()
    del ctrl.obsm["X_src"]
    with pytest.raises(ValueError, match=r"AnnData 0 of 2 holds source cells but lacks obsm/X_src \(source_rep_quad\)"):
        _manager().get_dataloaders([ctrl, pert], batch_size=8)


def test_the_linear_term_needs_one_width():
    ctrl, pert = _split_files()
    ctrl.obsm["K"] = np.ones((ctrl.n_obs, 4), "float32")
    pert.obsm["K"] = np.ones((pert.n_obs, 6), "float32")
    dm = DataManagerConfig(
        **{
            **_manager().config.model_dump(),
            "source_rep_quad": None,
            "target_rep_quad": None,
            "use_linear": True,
            "rep_lin": "K",
        }
    ).build()
    with pytest.raises(ValueError, match=r"obsm/K \(rep_lin\) is 4 wide in AnnData 0 \(sources\), 6 wide in AnnData 1"):
        dm.get_dataloaders([ctrl, pert], batch_size=8)


def test_the_state_may_differ_in_width_across_sides():
    # GENOT generates targets from noise conditioned on an encoded source, so the sides may live in different
    # spaces; whether the state must match is the method's call, not the loader's.
    ctrl, pert = _split_files()
    ctrl.obsm["K"] = np.ones((ctrl.n_obs, 4), "float32")
    pert.obsm["K"] = np.ones((pert.n_obs, 6), "float32")
    dm = DataManagerConfig(**{**_manager().config.model_dump(), "sample_rep": "K"}).build()
    batch = next(iter(dm.get_dataloaders([ctrl, pert], batch_size=8)["train"]))
    assert batch["source_state"].shape[-1] == 4 and batch["target_state"].shape[-1] == 6


def test_quadratic_terms_may_differ_in_width():
    ctrl, pert = _split_files()  # X_src is 4 wide, X_tgt 3 wide: Gromov-Wasserstein compares each side within itself
    batch = next(iter(_manager().get_dataloaders([ctrl, pert], batch_size=8)["train"]))
    assert batch["source_coupling_quad"].shape[-1] == 4 and batch["target_coupling_quad"].shape[-1] == 3


def test_datamodule_takes_each_sides_dims_from_its_files():
    ctrl, pert = _split_files()
    datamodule = FlowDataModule.from_adata([ctrl, pert], _manager(), batch_size=8, n_train_steps=2)
    dims = datamodule.data_dims
    assert (dims.source_quad_dim, dims.target_quad_dim) == (4, 3)
    batch = next(iter(datamodule.train_dataloader()))
    assert batch["source_coupling_quad"].shape[-1] == 4 and batch["target_coupling_quad"].shape[-1] == 3


def test_a_side_quad_needs_one_width_across_its_files():
    ctrl, pert = _split_files()
    other = pert.copy()
    other.obsm["X_tgt"] = np.ones((other.n_obs, 5), "float32")
    with pytest.raises(
        ValueError, match=r"target_quad_dim differs across the AnnData holding targets: AnnData 1: 3, AnnData 2: 5"
    ):
        _manager().get_data_dimensionalities([ctrl, pert, other])
