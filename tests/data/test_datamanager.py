import numpy as np
import pytest
from anndata import AnnData

from sc_flow._constants import ORIGINAL_INDEX_KEY
from sc_flow.data._composite import MatchedData, NestedData
from sc_flow.data._manager import DataManager
from sc_flow.data.containers._categorical import CategoricalData
from sc_flow.data.containers._coupling import CouplingData
from sc_flow.data.containers._distribution import DistributionData
from sc_flow.data.containers._mixed_type import MixedTypeData
from sc_flow.data.containers._state import StateData

from tests.data.conftest import (
    CELL_LINES,
    DRUGS,
    N_CELL_LINES,
    N_DRUGS,
    _make_obs,
)


def _make_manager(**overrides) -> DataManager:
    """DataManager with cell_line as group and drug as condition."""
    defaults = dict(
        conditions={"drug": ("drug",)},
        conditions_reps={"drug": "drug"},
        groups=("cell_line",),
        groups_reps={"cell_line": "cell_line"},
    )
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
