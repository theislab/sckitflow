import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from tests.data.conftest import (
    BATCHES,
    CELL_LINES,
    CELLS_PER_COMBINATION,
    DOSES,
    DRUGS,
    N_BATCHES,
    N_CELL_LINES,
    N_DOSES,
    N_DRUGS,
    N_PLATES,
    PLATES,
    _make_obs,
)

from sckitflow._constants import CONDITION_LEVEL_NAME, GROUP_LEVEL_NAME
from sckitflow.data.grouping._indexer import HierarchicalIndexer
from sckitflow.data.grouping._selector import IndexSelector


def _build(obs: pd.DataFrame, groups_cols, conditions_cols):
    """Create indexer, selector, index, and re-sorted obs.

    Re-sorts *obs* by exactly the used columns (groups then conditions)
    so that contiguous-slice logic works even when *obs* carries extra
    columns sorted by a wider set.
    """
    sort_cols = list(groups_cols or []) + list(conditions_cols or [])
    if sort_cols:
        obs = obs.sort_values(sort_cols).reset_index(drop=True)
    indexer = HierarchicalIndexer(groups_cols=groups_cols, conditions_cols=conditions_cols)
    selector = IndexSelector.init_from_indexer(indexer)
    index = indexer.create_index(obs)
    return selector, index, obs


class TestNestedDict:
    """index_to_nested_dict: correct tree structure, slices, and coverage."""

    def test_single_group_single_condition(self, adata_small: AnnData):
        selector, index, obs = _build(adata_small.obs, ["cell_line"], ["drug"])
        nd = selector.index_to_nested_dict(index)

        assert set(nd.mapping.keys()) == {(cl,) for cl in CELL_LINES}
        for node in nd.mapping.values():
            assert set(node.mapping.keys()) == {(d,) for d in DRUGS}

    def test_single_group_two_conditions(self, adata_small: AnnData):
        selector, index, obs = _build(adata_small.obs, ["cell_line"], ["dose", "drug"])
        nd = selector.index_to_nested_dict(index)

        hek = nd.mapping[("HEK293",)]
        expected_cond_keys = {(dose, drug) for dose in DOSES for drug in DRUGS}
        assert set(hek.mapping.keys()) == expected_cond_keys

    def test_two_groups_single_condition(self, adata_small: AnnData):
        selector, index, obs = _build(adata_small.obs, ["batch", "cell_line"], ["drug"])
        nd = selector.index_to_nested_dict(index)

        expected_group_keys = {(b, cl) for b in BATCHES for cl in CELL_LINES}
        assert set(nd.mapping.keys()) == expected_group_keys

        for node in nd.mapping.values():
            assert set(node.mapping.keys()) == {(d,) for d in DRUGS}
            for leaf in node.mapping.values():
                assert isinstance(leaf, slice)

    def test_three_groups_single_condition(self, adata_small: AnnData):
        groups = ["batch", "cell_line", "plate"]
        selector, index, obs = _build(adata_small.obs, groups, ["drug"])
        nd = selector.index_to_nested_dict(index)

        expected_group_keys = {(b, cl, p) for b in BATCHES for cl in CELL_LINES for p in PLATES}
        assert set(nd.mapping.keys()) == expected_group_keys

        for node in nd.mapping.values():
            assert set(node.mapping.keys()) == {(d,) for d in DRUGS}

    def test_no_grouping(self, adata_small: AnnData):
        selector, index, obs = _build(adata_small.obs, None, None)
        nd = selector.index_to_nested_dict(index)

        assert list(nd.mapping.keys()) == [()]
        inner = nd.mapping[()]
        assert list(inner.mapping.keys()) == [()]
        assert inner.mapping[()] == slice(0, len(obs))

    def test_group_only(self, adata_small: AnnData):
        selector, index, obs = _build(adata_small.obs, ["cell_line"], None)
        nd = selector.index_to_nested_dict(index)

        assert set(nd.mapping.keys()) == {(cl,) for cl in CELL_LINES}
        n_per_cell_line = len(obs) // N_CELL_LINES
        for i, cl in enumerate(sorted(CELL_LINES)):
            assert nd.mapping[(cl,)].mapping[()] == slice(
                i * n_per_cell_line,
                (i + 1) * n_per_cell_line,
            )

    def test_condition_only(self):
        """No groups -- build a minimal obs sorted by conditions only."""
        obs = _make_obs({"drug": DRUGS})
        selector, index, obs = _build(obs, None, ["drug"])
        nd = selector.index_to_nested_dict(index)

        assert list(nd.mapping.keys()) == [()]
        inner = nd.mapping[()]
        assert set(inner.mapping.keys()) == {(d,) for d in DRUGS}

    @pytest.mark.parametrize(
        "groups_cols,conditions_cols",
        [
            pytest.param(["cell_line"], ["drug"], id="1g_1c"),
            pytest.param(["cell_line"], ["dose", "drug"], id="1g_2c"),
            pytest.param(["batch", "cell_line"], ["drug"], id="2g_1c"),
            pytest.param(["batch", "cell_line", "plate"], ["drug"], id="3g_1c"),
            pytest.param(None, None, id="no_grouping"),
        ],
    )
    def test_slices_cover_all_rows_exactly_once(self, adata_small, groups_cols, conditions_cols):
        selector, index, obs = _build(adata_small.obs, groups_cols, conditions_cols)
        nd = selector.index_to_nested_dict(index)

        covered = np.zeros(len(obs), dtype=bool)
        for group_node in nd.mapping.values():
            for leaf in group_node.mapping.values():
                assert not covered[leaf].any(), "overlapping slices"
                covered[leaf] = True
        assert covered.all(), "some rows not covered"

    @pytest.mark.parametrize(
        "groups_cols,conditions_cols",
        [
            pytest.param(["cell_line"], ["dose", "drug"], id="1g_2c"),
            pytest.param(["batch", "cell_line"], ["drug"], id="2g_1c"),
            pytest.param(["batch", "cell_line", "plate"], ["drug"], id="3g_1c"),
        ],
    )
    def test_leaf_rows_match_key_values(self, adata_small, groups_cols, conditions_cols):
        """Every row inside a leaf slice has the column values that the key promises."""
        selector, index, obs = _build(adata_small.obs, groups_cols, conditions_cols)
        nd = selector.index_to_nested_dict(index)

        for g_key, group_node in nd.mapping.items():
            g_vals = g_key if isinstance(g_key, tuple) else (g_key,)
            for c_key, leaf in group_node.mapping.items():
                c_vals = c_key if isinstance(c_key, tuple) else (c_key,)
                rows = obs.iloc[leaf]
                for col, val in zip(groups_cols, g_vals, strict=True):
                    assert (rows[col] == val).all()
                for col, val in zip(conditions_cols, c_vals, strict=True):
                    assert (rows[col] == val).all()


class TestQueryLevel:
    """query_level_with_dict: correct filtering and error handling."""

    def test_query_group_level(self, adata_small: AnnData):
        selector, index, obs = _build(adata_small.obs, ["cell_line"], ["drug"])
        result = selector.query_level_with_dict(
            GROUP_LEVEL_NAME,
            {"cell_line": "HeLa"},
            index,
        )

        n_expected = len(obs) // N_CELL_LINES
        assert len(result) == n_expected
        pos = result.names.index((GROUP_LEVEL_NAME, "cell_line"))
        assert (result.get_level_values(pos) == "HeLa").all()

    def test_query_condition_single_col(self):
        obs = _make_obs({"drug": DRUGS})
        selector, index, obs = _build(obs, None, ["drug"])
        result = selector.query_level_with_dict(
            CONDITION_LEVEL_NAME,
            {"drug": "control"},
            index,
        )

        assert len(result) == CELLS_PER_COMBINATION
        pos = result.names.index((CONDITION_LEVEL_NAME, "drug"))
        assert (result.get_level_values(pos) == "control").all()

    def test_query_condition_multi_col(self):
        obs = _make_obs({"dose": DOSES, "drug": DRUGS})
        selector, index, obs = _build(obs, None, ["dose", "drug"])
        result = selector.query_level_with_dict(
            CONDITION_LEVEL_NAME,
            {"drug": "aspirin", "dose": "low"},
            index,
        )

        assert len(result) == CELLS_PER_COMBINATION
        pos_drug = result.names.index((CONDITION_LEVEL_NAME, "drug"))
        pos_dose = result.names.index((CONDITION_LEVEL_NAME, "dose"))
        assert (result.get_level_values(pos_drug) == "aspirin").all()
        assert (result.get_level_values(pos_dose) == "low").all()

    def test_query_condition_partial_filter(self):
        """Filtering on the leading condition column returns all matching rows."""
        obs = _make_obs({"dose": DOSES, "drug": DRUGS})
        selector, index, obs = _build(obs, None, ["dose", "drug"])
        result = selector.query_level_with_dict(
            CONDITION_LEVEL_NAME,
            {"dose": "low"},
            index,
        )

        expected_n = N_DRUGS * CELLS_PER_COMBINATION
        assert len(result) == expected_n
        pos = result.names.index((CONDITION_LEVEL_NAME, "dose"))
        assert (result.get_level_values(pos) == "low").all()

    def test_wrong_column_for_level_raises(self, adata_small: AnnData):
        selector, index, obs = _build(adata_small.obs, ["cell_line"], ["drug"])
        with pytest.raises(ValueError):
            selector.query_level_with_dict(CONDITION_LEVEL_NAME, {"cell_line": "HeLa"}, index)
        with pytest.raises(ValueError):
            selector.query_level_with_dict(GROUP_LEVEL_NAME, {"drug": "aspirin"}, index)


class TestQueryWithDict:
    """query_with_dict: chains group + condition filtering."""

    def test_filter_group_and_all_conditions(self, adata_small: AnnData):
        selector, index, obs = _build(adata_small.obs, ["cell_line"], ["dose", "drug"])
        result = selector.query_with_dict(
            {
                GROUP_LEVEL_NAME: {"cell_line": "HEK293"},
                CONDITION_LEVEL_NAME: {"dose": "low", "drug": "control"},
            },
            index,
        )

        n_expected = len(obs) // (N_CELL_LINES * N_DOSES * N_DRUGS)
        assert len(result) == n_expected
        pos_cl = result.names.index((GROUP_LEVEL_NAME, "cell_line"))
        pos_dose = result.names.index((CONDITION_LEVEL_NAME, "dose"))
        pos_drug = result.names.index((CONDITION_LEVEL_NAME, "drug"))
        assert (result.get_level_values(pos_cl) == "HEK293").all()
        assert (result.get_level_values(pos_dose) == "low").all()
        assert (result.get_level_values(pos_drug) == "control").all()

    def test_filter_group_only(self, adata_small: AnnData):
        selector, index, obs = _build(adata_small.obs, ["cell_line"], ["drug"])
        result = selector.query_with_dict(
            {GROUP_LEVEL_NAME: {"cell_line": "HeLa"}},
            index,
        )

        n_expected = len(obs) // N_CELL_LINES
        assert len(result) == n_expected
        pos = result.names.index((GROUP_LEVEL_NAME, "cell_line"))
        assert (result.get_level_values(pos) == "HeLa").all()

    def test_filter_two_groups_and_condition(self, adata_small: AnnData):
        selector, index, obs = _build(adata_small.obs, ["batch", "cell_line"], ["drug"])
        result = selector.query_with_dict(
            {
                GROUP_LEVEL_NAME: {"batch": "batch_1", "cell_line": "HeLa"},
                CONDITION_LEVEL_NAME: {"drug": "aspirin"},
            },
            index,
        )

        n_expected = len(obs) // (N_BATCHES * N_CELL_LINES * N_DRUGS)
        assert len(result) == n_expected
        pos_b = result.names.index((GROUP_LEVEL_NAME, "batch"))
        pos_cl = result.names.index((GROUP_LEVEL_NAME, "cell_line"))
        pos_d = result.names.index((CONDITION_LEVEL_NAME, "drug"))
        assert (result.get_level_values(pos_b) == "batch_1").all()
        assert (result.get_level_values(pos_cl) == "HeLa").all()
        assert (result.get_level_values(pos_d) == "aspirin").all()

    def test_filter_three_groups(self, adata_small: AnnData):
        groups = ["batch", "cell_line", "plate"]
        selector, index, obs = _build(adata_small.obs, groups, ["drug"])
        result = selector.query_with_dict(
            {
                GROUP_LEVEL_NAME: {
                    "batch": "batch_1",
                    "cell_line": "HEK293",
                    "plate": "plate_A",
                },
                CONDITION_LEVEL_NAME: {"drug": "aspirin"},
            },
            index,
        )

        n_expected = len(obs) // (N_BATCHES * N_CELL_LINES * N_PLATES * N_DRUGS)
        assert len(result) == n_expected
        for col, val in [("batch", "batch_1"), ("cell_line", "HEK293"), ("plate", "plate_A")]:
            pos = result.names.index((GROUP_LEVEL_NAME, col))
            assert (result.get_level_values(pos) == val).all()
        pos_d = result.names.index((CONDITION_LEVEL_NAME, "drug"))
        assert (result.get_level_values(pos_d) == "aspirin").all()

    def test_invalid_level_raises(self, adata_small: AnnData):
        selector, index, obs = _build(adata_small.obs, ["cell_line"], ["drug"])
        with pytest.raises(ValueError):
            selector.query_with_dict(
                {
                    CONDITION_LEVEL_NAME: {"drug": "control"},
                    "bogus_level": {"cell_line": "HeLa"},
                },
                index,
            )
