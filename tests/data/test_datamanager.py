import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from sckitflow.data._manager import DataManager
from sckitflow.data.containers._categorical import CategoricalData
from sckitflow.data.containers._coupling import CouplingData
from sckitflow.data.containers._distribution import DistributionData
from sckitflow.data.containers._mixed_type import MixedTypeData
from sckitflow.data.containers._state import StateData


def _make_manager(**overrides) -> DataManager:
    """DataManager with cell_line as group and drug as condition."""
    defaults = {
        "conditions": {"drug": ("drug",)},
        "conditions_reps": {"drug": "drug"},
        "groups": ("cell_line",),
        "groups_reps": {"cell_line": "cell_line"},
    }
    defaults.update(overrides)
    return DataManager(**defaults)


def _make_manager_with_continuous(**overrides):
    """DataManager with a continuous condition covariate (X_repr) and categorical drug condition."""
    defaults = {
        "conditions": {"drug": ("drug",)},
        "conditions_reps": {"drug": "drug"},
        "conditions_covariates": ["X_repr"],  # continuous covariate from obsm
        "groups": ("cell_line",),
        "groups_reps": {"cell_line": "cell_line"},
    }
    defaults.update(overrides)
    return DataManager(**defaults)


class TestDistributionData:
    """get_distribution_data: building DistributionData from AnnData."""

    def test_types_and_lengths(self, adata_small: AnnData):
        manager = _make_manager()
        distr = manager.get_distribution_data(adata_small)

        assert isinstance(distr, DistributionData)
        assert len(distr) == adata_small.n_obs
        assert isinstance(distr.state_data, StateData)
        assert isinstance(distr.groups_data, CategoricalData)
        assert len(distr.state_data) == adata_small.n_obs
        assert len(distr.groups_data) == adata_small.n_obs
        if distr.condition_data is not None:
            assert isinstance(distr.condition_data, MixedTypeData)
            assert len(distr.condition_data) == adata_small.n_obs

    def test_slicing(self, adata_small: AnnData):
        manager = _make_manager()
        distr = manager.get_distribution_data(adata_small)
        n_slice = 5
        idxs = np.arange(n_slice)
        sliced = distr[idxs]

        assert len(sliced) == n_slice
        for attr in ("state_data", "groups_data", "condition_data", "response_data"):
            val = getattr(sliced, attr)
            if val is not None:
                assert len(val) == n_slice

    def test_empty_slice(self, adata_small: AnnData):
        manager = _make_manager()
        distr = manager.get_distribution_data(adata_small)
        sliced = distr[np.array([], dtype=int)]

        assert len(sliced) == 0
        for attr in ("state_data", "groups_data", "condition_data", "response_data"):
            val = getattr(sliced, attr)
            if val is not None:
                assert len(val) == 0

    def test_repr(self, adata_small: AnnData):
        manager = _make_manager()
        distr = manager.get_distribution_data(adata_small)
        r = repr(distr)
        assert "DistributionData" in r
        assert f"n_obs={adata_small.n_obs}" in r

    def test_coupling_defaults_to_state(self, adata_small: AnnData):
        manager = _make_manager()
        distr = manager.get_distribution_data(adata_small)
        assert isinstance(distr.source_coupling_data, CouplingData)
        assert isinstance(distr.target_coupling_data, CouplingData)
        assert len(distr.source_coupling_data) == adata_small.n_obs
        assert len(distr.target_coupling_data) == adata_small.n_obs


class TestRequireTargetState:
    """get_distribution_data with require_target_state=False (predict without target states)."""

    def test_get_distribution_data_state_is_none(self, adata_small: AnnData):
        manager = _make_manager()
        distr = manager.get_distribution_data(adata_small, require_target_state=False)

        assert distr.state_data is None
        assert len(distr) == adata_small.n_obs
        assert isinstance(distr.groups_data, CategoricalData)

    def test_get_distribution_data_works_without_x(self):
        """An AnnData with only `.obs` (no `.X`) can be compiled when require_target_state=False."""
        from tests.data.conftest import CELL_LINES, DRUGS

        obs = pd.DataFrame(
            {"cell_line": [CELL_LINES[0]] * 3, "drug": [DRUGS[0]] * 3},
        ).astype("category")
        ad = AnnData(obs=obs)
        # condition/group reps are looked up by value in `.uns`, regardless of `.X`
        ad.uns["drug"] = {DRUGS[0]: np.zeros((1, 4))}
        ad.uns["cell_line"] = {CELL_LINES[0]: np.zeros((1, 4))}

        manager = _make_manager()
        distr = manager.get_distribution_data(ad, require_target_state=False)

        assert distr.state_data is None
        assert len(distr) == 3


class TestConditionSpaceView:
    def test_get_distribution_data_with_condition_space(self, adata_small: AnnData):
        manager = _make_manager_with_continuous(condition_state_key="X_repr")
        # ensure X_repr exists in obsm
        if "X_repr" not in adata_small.obsm:
            adata_small.obsm["X_repr"] = np.random.randn(adata_small.n_obs, 10)

        distr = manager.get_distribution_data(adata_small)
        # state_data becomes the continuous covariate
        assert isinstance(distr.state_data, StateData)
        np.testing.assert_array_equal(distr.state_data.X, adata_small.obsm["X_repr"])

        # condition_data still exists (categorical part remains)
        assert distr.condition_data is not None
        # The continuous key "X_repr" should be removed. Since it was the only continuous key,
        # continuous_covariates becomes None. That's acceptable.
        if distr.condition_data.continuous_covariates is not None:
            assert "X_repr" not in distr.condition_data.continuous_covariates.mapping
        # categorical part remains
        assert distr.condition_data.categorical_covariates is not None

        # coupling data reinitialized
        assert distr.source_coupling_data is not None
        assert distr.target_coupling_data is not None
        assert len(distr.source_coupling_data) == len(distr.state_data)

    def test_get_distribution_data_invalid_condition_state_key(self, adata_small: AnnData):
        # Use a manager without continuous covariates -- only categorical conditions.
        manager = _make_manager(condition_state_key="invalid_key")
        with pytest.raises(KeyError, match="Key invalid_key not found"):
            manager.get_distribution_data(adata_small)

    def test_get_data_dimensionalities_with_condition_space(self, adata_small: AnnData):
        manager = _make_manager_with_continuous(condition_state_key="X_repr")
        if "X_repr" not in adata_small.obsm:
            adata_small.obsm["X_repr"] = np.random.randn(adata_small.n_obs, 10)

        dims = manager.get_data_dimensionalities(adata_small)
        # state dimension comes from continuous covariate
        assert dims.state_dim == adata_small.obsm["X_repr"].shape[1]
        # condition has a categorical part (drug) so categorical dim should be present
        assert dims.condition_reps_dims is not None and all(d > 0 for d in dims.condition_reps_dims.values())
        # the continuous covariate was consumed, so continuous condition dim should be 0 or None
        assert dims.condition_continuous_dims == {}


def _with_split(adata: AnnData) -> AnnData:
    """Attach a 'split' column: controls -> 'control', perturbed groups split train/val by cell line."""
    adata = adata.copy()
    is_ctrl = adata.obs["drug"].astype(str).to_numpy() == "control"
    lines = adata.obs["cell_line"].astype(str).to_numpy()
    val_line = sorted(set(lines))[0]
    split = np.where(is_ctrl, "control", np.where(lines == val_line, "val", "train"))
    adata.obs["split"] = pd.Categorical(split)
    return adata


class TestGetDataloaders:
    """get_dataloaders: one streaming Loader per split, controls shared."""

    def test_one_loader_per_split(self, adata_small: AnnData):
        ad = _with_split(adata_small)
        dm = _make_manager(control_values_dict={"drug": "control"})
        loaders = dm.get_dataloaders(ad, split_by="split", to="torch", batch_size=8)

        # controls are the shared source, never their own split
        assert set(loaders) == {"train", "val"}
        for loader in loaders.values():
            assert len(loader) >= 1
            step_data = next(iter(loader))
            assert step_data["target_state"] is not None
            assert step_data["source_state"] is not None  # matched controls

    def test_missing_split_column_raises(self, adata_small: AnnData):
        dm = _make_manager()
        with pytest.raises(KeyError, match="not found"):
            dm.get_dataloaders(adata_small, split_by="does_not_exist")


class TestGetEvalLoader:
    """get_eval_loader: deterministic per-group (StepData, leaf) for prediction."""

    def _n_groups(self, adata: AnnData, *, exclude_control: bool) -> int:
        obs = adata.obs[["cell_line", "drug"]].astype(str)
        if exclude_control:
            obs = obs[obs["drug"] != "control"]
        return obs.drop_duplicates().shape[0]

    def test_paired_predicts_noncontrol_matched_to_controls(self, adata_small: AnnData):
        dm = _make_manager(control_values_dict={"drug": "control"})
        el = dm.get_eval_loader(adata_small, to="torch")

        assert len(el) == self._n_groups(adata_small, exclude_control=True)
        assert el.group_cols == ("cell_line", "drug")
        for step_data, leaf in el:
            assert step_data["target_state"] is not None
            assert step_data["source_state"] is not None  # matched controls
            assert "control" not in leaf  # only perturbed groups are predicted

    def test_max_per_group_one_dedups_and_reiterates(self, adata_small: AnnData):
        dm = _make_manager(control_values_dict={"drug": "control"})
        el = dm.get_eval_loader(adata_small, max_per_group=1, to="torch")

        batches = list(el)
        assert all(step_data["target_state"].shape[0] == 1 for step_data, _ in batches)
        assert len({leaf for _, leaf in batches}) == len(batches)  # one per group
        assert len(list(el)) == len(el)  # re-iterable

    def test_unpaired_predicts_all_groups_without_source(self, adata_small: AnnData):
        dm = _make_manager()  # no control_values_dict
        el = dm.get_eval_loader(adata_small, to="torch")

        assert len(el) == self._n_groups(adata_small, exclude_control=False)
        step_data, _ = next(iter(el))
        assert step_data["source_state"] is None  # unpaired: no control link

    def test_control_values_override_makes_it_paired(self, adata_small: AnnData):
        dm = _make_manager()  # instance is unpaired
        el = dm.get_eval_loader(adata_small, control_values_dict={"drug": "control"}, to="torch")

        assert len(el) == self._n_groups(adata_small, exclude_control=True)
        for step_data, leaf in el:
            assert "control" not in leaf
            assert step_data["source_state"] is not None
