import numpy as np
import pytest
from anndata import AnnData
from tests.data.conftest import (
    N_CELL_LINES,
    N_DRUGS,
)

from sc_flow._constants import ORIGINAL_INDEX_KEY
from sc_flow.data._composite import MatchedData, NestedData
from sc_flow.data._manager import DataManager
from sc_flow.data.containers._categorical import CategoricalData
from sc_flow.data.containers._coupling import CouplingData
from sc_flow.data.containers._distribution import DistributionData
from sc_flow.data.containers._mixed_type import MixedTypeData
from sc_flow.data.containers._state import StateData


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


class TestCompileAdata:
    """compile_adata: sorting, validation, and nested output."""

    def test_sort_produces_nested_data(self, adata_small: AnnData):
        manager = _make_manager()
        nested = manager.compile_adata(adata_small, sort=True)
        assert isinstance(nested, NestedData)

    def test_sort_leaves_are_matched(self, adata_small: AnnData):
        manager = _make_manager()
        nested = manager.compile_adata(adata_small, sort=True)

        def check(node):
            if isinstance(node, NestedData):
                for v in node.mapping.values():
                    check(v)
            else:
                assert isinstance(node, MatchedData)
                assert len(node.target) > 0

        check(nested)

    def test_leaf_count_matches_combinations(self, adata_small: AnnData):
        """Number of leaves equals |groups| * |conditions|."""
        manager = _make_manager()
        nested = manager.compile_adata(adata_small, sort=True)

        leaves = []

        def collect(node):
            if isinstance(node, NestedData):
                for v in node.mapping.values():
                    collect(v)
            else:
                leaves.append(node)

        collect(nested)
        assert len(leaves) == N_CELL_LINES * N_DRUGS

    def test_unsorted_without_sort_raises(self, adata_small: AnnData):
        rng = np.random.default_rng(0)
        shuffled = adata_small[rng.permutation(adata_small.n_obs)].copy()

        manager = _make_manager()
        with pytest.raises(ValueError, match="not sorted"):
            manager.compile_adata(shuffled, sort=False)

    def test_presorted_without_sort_succeeds(self, adata_small: AnnData):
        manager = _make_manager()
        sorted_ad = manager.sort_adata(adata_small)
        nested = manager.compile_adata(sorted_ad, sort=False)
        assert isinstance(nested, NestedData)


class TestSortAdata:
    """sort_adata: returns sorted copy, preserves original index."""

    def test_preserves_original_index(self, adata_small: AnnData):
        manager = _make_manager()
        original_names = adata_small.obs_names.copy()
        sorted_ad = manager.sort_adata(adata_small)

        assert ORIGINAL_INDEX_KEY in sorted_ad.obs.columns
        assert len(sorted_ad) == adata_small.n_obs
        assert set(sorted_ad.obs[ORIGINAL_INDEX_KEY]) == set(original_names)

    def test_does_not_mutate_input(self, adata_small: AnnData):
        original_names = adata_small.obs_names.copy()
        manager = _make_manager()
        _ = manager.sort_adata(adata_small)
        assert (adata_small.obs_names == original_names).all()

    def test_result_is_sorted(self, adata_small: AnnData):
        manager = _make_manager()
        sorted_ad = manager.sort_adata(adata_small)
        distr = manager.get_distribution_data(sorted_ad)
        assert distr.is_sorted

    def test_shuffle_then_sort_is_sorted(self, adata_small: AnnData):
        rng = np.random.default_rng(7)
        shuffled = adata_small[rng.permutation(adata_small.n_obs)].copy()

        manager = _make_manager()
        sorted_ad = manager.sort_adata(shuffled)
        distr = manager.get_distribution_data(sorted_ad)
        assert distr.is_sorted

    def test_noop_without_columns(self, adata_small: AnnData):
        manager = DataManager()
        sorted_ad = manager.sort_adata(adata_small)
        assert (sorted_ad.obs_names == adata_small.obs_names).all()


class TestSourceKey:
    def test_with_control_values(self):
        manager = DataManager(
            conditions={"drug": ("drug",)},
            conditions_reps={"drug": "drug"},
            control_values_dict={"drug": "control"},
        )
        key = manager.source_key
        assert isinstance(key, tuple)
        assert len(key) > 0

    def test_none_without_control_values(self):
        manager = _make_manager()
        assert manager.source_key is None


class TestConditionSpaceView:
    def test_get_distribution_data_with_condition_space(self, adata_small: AnnData):
        manager = _make_manager_with_continuous()
        # ensure X_repr exists in obsm
        if "X_repr" not in adata_small.obsm:
            adata_small.obsm["X_repr"] = np.random.randn(adata_small.n_obs, 10)

        distr = manager.get_distribution_data(adata_small, view_on_condition_space=True, condition_state_key="X_repr")
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
        # Use a manager without continuous covariates – only categorical conditions.
        manager = _make_manager()  # from the fixture, no `conditions_covariates`
        # The condition_data has only categorical covariates (drug).
        with pytest.raises(KeyError, match="Key invalid_key not found"):
            manager.get_distribution_data(adata_small, view_on_condition_space=True, condition_state_key="invalid_key")

    def test_get_data_dimensionalities_with_condition_space(self, adata_small: AnnData):
        manager = _make_manager_with_continuous()
        if "X_repr" not in adata_small.obsm:
            adata_small.obsm["X_repr"] = np.random.randn(adata_small.n_obs, 10)

        dims = manager.get_data_dimensionalities(
            adata_small, view_on_condition_space=True, condition_state_key="X_repr"
        )
        # state dimension comes from continuous covariate
        assert dims.state_dim == adata_small.obsm["X_repr"].shape[1]
        # condition has a categorical part (drug) so categorical dim should be present
        assert dims.condition_reps_dims is not None and all(d > 0 for d in dims.condition_reps_dims.values())
        # the continuous covariate was consumed, so continuous condition dim should be 0 or None
        assert dims.condition_continuous_dims == {}


class TestConditionSpacePairedSettings:
    """Test the allow_paired_settings_on_condition_view flag."""

    def test_paired_disabled_by_default_on_condition_view(self, adata_small: AnnData):
        """When view_on_condition_space=True and allow_paired_settings_on_condition_view=False (default),
        the source_key / matched_keys are ignored, so no source distribution appears in MatchedData."""
        # Add continuous covariate
        if "X_repr" not in adata_small.obsm:
            adata_small.obsm["X_repr"] = np.random.randn(adata_small.n_obs, 10)

        # Create manager with control values (paired setting) but allow_paired_settings_on_condition_view=False
        manager = DataManager(
            conditions={"drug": ("drug",)},
            conditions_reps={"drug": "drug"},
            conditions_covariates=["X_repr"],
            groups=("cell_line",),
            groups_reps={"cell_line": "cell_line"},
            control_values_dict={"drug": "control"},
            allow_paired_settings_on_condition_view=False,
        )
        # Compile with view_on_condition_space=True
        nested = manager.compile_adata(
            adata_small, sort=True, view_on_condition_space=True, condition_state_key="X_repr"
        )

        # Flatten and check that none of the MatchedData nodes have a source distribution
        def check_no_source(node):
            if isinstance(node, NestedData):
                for v in node.mapping.values():
                    check_no_source(v)
            else:
                assert node.source is None

        check_no_source(nested)

    def test_paired_enabled_on_condition_view(self, adata_small: AnnData):
        """When allow_paired_settings_on_condition_view=True, pairing works as usual."""
        if "X_repr" not in adata_small.obsm:
            adata_small.obsm["X_repr"] = np.random.randn(adata_small.n_obs, 10)

        manager = DataManager(
            conditions={"drug": ("drug",)},
            conditions_reps={"drug": "drug"},
            conditions_covariates=["X_repr"],
            groups=("cell_line",),
            groups_reps={"cell_line": "cell_line"},
            control_values_dict={"drug": "control"},
            allow_paired_settings_on_condition_view=True,
        )
        nested = manager.compile_adata(
            adata_small, sort=True, view_on_condition_space=True, condition_state_key="X_repr"
        )
        # Check that some MatchedData nodes have source
        has_source = False

        def check_source_present(node):
            nonlocal has_source
            if isinstance(node, NestedData):
                for v in node.mapping.values():
                    check_source_present(v)
            else:
                if node.source is not None:
                    has_source = True

        check_source_present(nested)
        assert has_source

    def test_paired_setting_without_condition_view_ignores_flag(self, adata_small: AnnData):
        """When view_on_condition_space=False, the flag has no effect; pairing works."""
        manager = DataManager(
            conditions={"drug": ("drug",)},
            conditions_reps={"drug": "drug"},
            groups=("cell_line",),
            groups_reps={"cell_line": "cell_line"},
            control_values_dict={"drug": "control"},
            allow_paired_settings_on_condition_view=False,  # should be ignored
        )
        nested = manager.compile_adata(adata_small, sort=True, view_on_condition_space=False)
        has_source = False

        def check_source(node):
            nonlocal has_source
            if isinstance(node, NestedData):
                for v in node.mapping.values():
                    check_source(v)
            else:
                if node.source is not None:
                    has_source = True

        check_source(nested)
        assert has_source  # pairing still active
