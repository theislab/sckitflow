import numpy as np
import pandas as pd
import pytest

from sc_flow.core.data._composite import MatchedData, NestedData
from sc_flow.core.data._mixins import BatchMixin, MappedLevelIndex
from sc_flow.core.data.containers._categorical import CategoricalData
from sc_flow.core.data.containers._distribution import DistributionData
from sc_flow.core.data.containers._mixed_type import MixedTypeData
from sc_flow.core.data.containers._state import StateData

from .shared import make_distribution


def make_distribution_with_continuous_condition(n=10, n_cont=3):
    """Create a DistributionData that has continuous condition covariates."""
    state = StateData(np.random.randn(n, 5))
    # Create a continuous condition covariate
    cont_mapping = {"X_cont": np.random.randn(n, n_cont)}
    continuous_covs = BatchMixin(cont_mapping)
    # Dummy categorical condition (optional, but to satisfy MixedTypeData)
    cat_df = pd.DataFrame({"dummy": ["A"] * n})
    cat_data = CategoricalData.from_pandas(cat_df)
    condition_data = MixedTypeData(
        categorical_covariates=cat_data,
        continuous_covariates=continuous_covs,
    )
    return DistributionData(state_data=state, condition_data=condition_data)


class TestMatchedData:
    def test_properties_with_source(self):
        target = make_distribution(n=10)
        source = make_distribution(n=5)
        matched = MatchedData(target_distribution=target, source_distribution=source)

        assert matched.target is target
        assert matched.source is source
        assert matched.n_target_obs == 10
        assert matched.n_source_obs == 5
        assert matched.n_tgt_obs == 10
        assert matched.n_src_obs == 5
        assert matched.target_distr is target
        assert matched.source_distr is source

    def test_properties_without_source(self):
        target = make_distribution(n=8)
        matched = MatchedData(target_distribution=target)

        assert matched.target is target
        assert matched.source is None
        assert matched.n_target_obs == 8
        assert matched.n_source_obs is None
        assert matched.n_tgt_obs == 8
        assert matched.n_src_obs is None


class TestNestedData:
    @pytest.fixture
    def leaf_index(self):
        return MappedLevelIndex(mapping={("a",): slice(0, 2), ("b",): slice(2, 4), ("c",): slice(4, 6)})

    # ---------- Existing tests (adjusted) ----------
    def test_init_leaf_node_without_source(self, leaf_index):
        data = make_distribution(n=10)
        nested = NestedData._init_leaf_node(data, leaf_index)

        assert isinstance(nested, NestedData)
        assert all(isinstance(v, MatchedData) for v in nested.mapping.values())
        for v in nested.mapping.values():
            assert v.source is None

    def test_init_leaf_node_with_source(self, leaf_index):
        data = make_distribution(n=10)
        source_key = ("a",)
        nested = NestedData._init_leaf_node(data, leaf_index, source_key=source_key)

        assert isinstance(nested, NestedData)
        # The source key itself is not included as a target, so all entries have a source
        for key, v in nested.mapping.items():
            assert isinstance(v, MatchedData)
            if key != source_key:  # source key is excluded, but we don't have it in mapping
                assert v.source is not None

    def test_init_tree_recursive(self):
        leaf_index_a = MappedLevelIndex(mapping={("leaf1",): slice(0, 2)})
        leaf_index_b = MappedLevelIndex(mapping={("leaf2",): slice(2, 4)})
        mapped_index = MappedLevelIndex(mapping={("a",): leaf_index_a, ("b",): leaf_index_b})

        data = make_distribution(n=10)
        nested = NestedData._init_tree(data, mapped_index)

        assert isinstance(nested, NestedData)
        for v in nested.mapping.values():
            assert isinstance(v, NestedData)
            for m in v.mapping.values():
                assert isinstance(m, MatchedData)

    def test_init_leaf_node_one_to_one(self, leaf_index):
        data = make_distribution(n=10)
        matched_keys = {("a",): ("b",), ("b",): ("c",)}
        nested = NestedData._init_leaf_node_one_to_one(data, leaf_index, matched_keys)

        assert isinstance(nested, NestedData)
        assert len(nested.mapping) == 2

        key_ab = (("a",), ("b",))
        matched_ab = nested.mapping[key_ab]
        # Compare n_obs and a representative attribute instead of full equality
        target_b = data[leaf_index.mapping[("b",)]]
        source_a = data[leaf_index.mapping[("a",)]]
        assert len(matched_ab.target) == len(target_b)
        assert len(matched_ab.source) == len(source_a)
        # Optionally check that the state data is the same (first row)
        # Here we assume state data is a numpy array; check first element.
        assert matched_ab.target.state_data.X[0] == target_b.state_data.X[0]

        key_bc = (("b",), ("c",))
        matched_bc = nested.mapping[key_bc]
        target_c = data[leaf_index.mapping[("c",)]]
        source_b = data[leaf_index.mapping[("b",)]]
        assert len(matched_bc.target) == len(target_c)
        assert len(matched_bc.source) == len(source_b)
        assert matched_bc.target.state_data.X[0] == target_c.state_data.X[0]

    def test_init_from_data_with_matched_keys_flat(self, leaf_index):
        """Wrap the leaf inside a root index so that init_from_data works."""
        data = make_distribution(n=10)
        # Create a root node containing the leaf
        root_index = MappedLevelIndex(mapping={(): leaf_index})
        matched_keys = {("a",): ("b",), ("b",): ("c",)}
        nested = NestedData.init_from_data(data, root_index, matched_keys=matched_keys)
        # The result should have one top-level key ((),) containing the actual data
        assert isinstance(nested, NestedData)
        assert len(nested.mapping) == 1
        actual = nested.mapping[()]
        assert len(actual.mapping) == 2
        assert (("a",), ("b",)) in actual.mapping
        assert (("b",), ("c",)) in actual.mapping

    def test_init_from_data_with_matched_keys_and_source_key_ignored(self, leaf_index):
        """When matched_keys is provided, source_key should be ignored."""
        data = make_distribution(n=10)
        root_index = MappedLevelIndex(mapping={(): leaf_index})
        matched_keys = {("a",): ("b",)}
        nested = NestedData.init_from_data(data, root_index, source_key=("c",), matched_keys=matched_keys)
        actual = nested.mapping[()]
        assert len(actual.mapping) == 1
        key = (("a",), ("b",))
        assert key in actual.mapping
        # Compare scalar attributes
        source_a = data[leaf_index.mapping[("a",)]]
        source_c = data[leaf_index.mapping[("c",)]]
        assert len(actual.mapping[key].source) == len(source_a)
        assert actual.mapping[key].source.state_data.X[0] == source_a.state_data.X[0]
        # Ensure source_key=("c",) was ignored by checking against slice contents
        assert len(actual.mapping[key].source) != len(source_c)
        assert actual.mapping[key].source.state_data.X[0] != source_c.state_data.X[0]

    def test_init_tree_with_matched_keys_nested(self):
        """Test a nested tree with one branch only (to avoid missing keys)."""
        leaf = MappedLevelIndex(mapping={("a",): slice(0, 2), ("b",): slice(2, 4)})
        root = MappedLevelIndex(mapping={("X",): leaf})

        data = make_distribution(n=10)
        matched_keys = {("a",): ("b",)}

        nested = NestedData._init_tree(data, root, matched_keys=matched_keys)

        assert set(nested.mapping.keys()) == {("X",)}
        sub = nested.mapping[("X",)]
        assert len(sub.mapping) == 1
        assert (("a",), ("b",)) in sub.mapping
        pair = sub.mapping[(("a",), ("b",))]
        target_b = data[leaf.mapping[("b",)]]
        source_a = data[leaf.mapping[("a",)]]
        assert len(pair.target) == len(target_b)
        assert len(pair.source) == len(source_a)
        assert pair.target.state_data.X[0] == target_b.state_data.X[0]


class TestMatchedDataAlign:
    """Test MatchedData.align method."""

    def test_no_continuous_covariates_returns_self(self):
        """If target has no continuous condition covariates, align returns self."""
        target = make_distribution(n=10)  # no condition_data
        source = make_distribution(n=5)
        matched = MatchedData(target_distribution=target, source_distribution=source)
        aligned = matched.align()
        assert aligned is matched

    def test_source_none_returns_self(self):
        """If source is None, align returns self."""
        target = make_distribution_with_continuous_condition(n=10)
        matched = MatchedData(target_distribution=target, source_distribution=None)
        aligned = matched.align()
        assert aligned is matched

    def test_equal_obs_returns_self(self):
        """If n_target_obs == n_source_obs, return self."""
        target = make_distribution_with_continuous_condition(n=10)
        source = make_distribution(n=10)  # source can be without continuous condition
        matched = MatchedData(target_distribution=target, source_distribution=source)
        aligned = matched.align()
        assert aligned is matched

    def test_target_shorter_slices_source(self):
        """If target has fewer observations, source should be sliced."""
        target = make_distribution_with_continuous_condition(n=6)
        source = make_distribution(n=10)
        matched = MatchedData(target_distribution=target, source_distribution=source)
        aligned = matched.align()
        # Sliced source should have 6 observations
        assert len(aligned.source) == 6
        # Check that the sliced source data matches the first 6 rows of original source
        np.testing.assert_array_equal(aligned.source.state_data.X, source.state_data.X[:6])
        # Target unchanged
        assert aligned.target is target

    def test_target_longer_repeats_source(self):
        """If target has more observations, source should be repeated."""
        target = make_distribution_with_continuous_condition(n=23)  # 23 observations
        source = make_distribution(n=5)  # 5 observations
        matched = MatchedData(target_distribution=target, source_distribution=source)
        aligned = matched.align()
        # Expected source length = 23 (repeated: 4 full repeats + remainder of 3)
        assert len(aligned.source) == 23
        # Check the concatenated array: should be source repeated 4 times + first 3 rows of source
        expected = np.vstack([source.state_data.X] * 4 + [source.state_data.X[:3]])
        np.testing.assert_array_equal(aligned.source.state_data.X, expected)

    def test_target_longer_multiple_of_source(self):
        """When target length is exact multiple of source length."""
        target = make_distribution_with_continuous_condition(n=20)
        source = make_distribution(n=5)
        matched = MatchedData(target_distribution=target, source_distribution=source)
        aligned = matched.align()
        assert len(aligned.source) == 20
        expected = np.vstack([source.state_data.X] * 4)
        np.testing.assert_array_equal(aligned.source.state_data.X, expected)

    def test_target_shorter_exact_slice(self):
        """When target length is less than source length (exact match not needed)."""
        target = make_distribution_with_continuous_condition(n=3)
        source = make_distribution(n=10)
        matched = MatchedData(target_distribution=target, source_distribution=source)
        aligned = matched.align()
        assert len(aligned.source) == 3
        np.testing.assert_array_equal(aligned.source.state_data.X, source.state_data.X[:3])
