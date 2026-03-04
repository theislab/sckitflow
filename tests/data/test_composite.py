import pandas as pd
import pytest

from sc_flow.data._composite import MatchedData, NestedData
from sc_flow.data._mixins import MappedLevelIndex

from .shared import make_distribution


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
    def mapped_index_leaf(self):
        return MappedLevelIndex(mapping={("a",): slice(0, 2), ("b",): slice(2, 4)})

    def test_init_leaf_node_without_source(self, mapped_index_leaf):
        data = make_distribution(n=10)
        nested = NestedData._init_leaf_node(data, mapped_index_leaf)

        assert isinstance(nested, NestedData)
        assert all(isinstance(v, MatchedData) for v in nested.mapping.values())
        for v in nested.mapping.values():
            assert v.source is None

    def test_init_leaf_node_with_source(self, mapped_index_leaf):
        data = make_distribution(n=10)
        source_key = ("a",)
        nested = NestedData._init_leaf_node(data, mapped_index_leaf, source_key=source_key)

        assert isinstance(nested, NestedData)
        for v in nested.mapping.values():
            assert isinstance(v, MatchedData)
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
