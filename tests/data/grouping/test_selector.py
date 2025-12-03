from collections.abc import Collection
from typing import Any

import pandas as pd
import pytest
from anndata import AnnData

from sc_flow._constants import CONDITION_LEVEL_NAME, GROUP_LEVEL_NAME
from sc_flow._types import NestedMappedLevelIndex
from sc_flow.data.grouping._indexer import HierarchicalIndexer
from sc_flow.data.grouping._selector import IndexSelector

inval_key = "wrong_key"


class TestIndexSelector:
    @pytest.mark.parametrize("groups_cols", [None, ["source_split"]])
    @pytest.mark.parametrize("conditions_cols", [None, ["drugA"], ["drugA", "drugB"]])
    @pytest.mark.parametrize("level_name", [CONDITION_LEVEL_NAME, GROUP_LEVEL_NAME])
    @pytest.mark.parametrize(
        "query_dict",
        [
            {
                "drugA": ("control",),
                "drugB": ("control",),
            },
            {
                "drugA": ("control",),
            },
            {
                "source_split": ("source_split0",),
            },
            {
                "drugA": ("control",),
                "source_split": ("source_split0",),
            },
        ],
    )
    def test_query_level_with_dict(
        self,
        adata: AnnData,
        groups_cols: Collection[str] | None,
        conditions_cols: Collection[str] | None,
        level_name: str,
        query_dict: dict[str, Any],
    ) -> None:
        # initialize indexer and selector
        indexer = HierarchicalIndexer(
            groups_cols=groups_cols,
            conditions_cols=conditions_cols,
        )
        selector = IndexSelector.init_from_indexer(indexer)
        index = indexer.create_index(adata.obs)

        # query index (fail cases)
        if level_name == CONDITION_LEVEL_NAME and "source_split" in query_dict:
            with pytest.raises(ValueError):
                query_res = selector.query_level_with_dict(
                    level_name,
                    query_dict,
                    index,
                )
            return None
        elif level_name == GROUP_LEVEL_NAME and "drugA" in query_dict:
            with pytest.raises(ValueError):
                query_res = selector.query_level_with_dict(
                    level_name,
                    query_dict,
                    index,
                )
            return None
        elif level_name == CONDITION_LEVEL_NAME and conditions_cols is None:
            with pytest.raises(ValueError):
                query_res = selector.query_level_with_dict(
                    level_name,
                    query_dict,
                    index,
                )
            return None
        elif level_name == CONDITION_LEVEL_NAME and len(set(query_dict).difference(set(conditions_cols))) > 0:
            with pytest.raises(ValueError):
                query_res = selector.query_level_with_dict(
                    level_name,
                    query_dict,
                    index,
                )
            return None
        elif level_name == GROUP_LEVEL_NAME and groups_cols is None:
            with pytest.raises(ValueError):
                query_res = selector.query_level_with_dict(
                    level_name,
                    query_dict,
                    index,
                )
            return
        query_res = selector.query_level_with_dict(
            level_name,
            query_dict,
            index,
        )
        assert isinstance(query_res, pd.MultiIndex)

    @pytest.mark.parametrize("groups_cols", [None, ["source_split"]])
    @pytest.mark.parametrize("conditions_cols", [None, ["drugA", "drugB"]])
    @pytest.mark.parametrize(
        "query_dict",
        [
            {
                CONDITION_LEVEL_NAME: {
                    "drugA": ("control",),
                    "drugB": ("control",),
                },
            },
            {
                CONDITION_LEVEL_NAME: {
                    "drugA": ("control",),
                },
            },
            {
                GROUP_LEVEL_NAME: {
                    "source_split": ("source_split0",),
                },
            },
            {
                CONDITION_LEVEL_NAME: {
                    "drugA": ("control",),
                    "drugB": ("control",),
                },
                GROUP_LEVEL_NAME: {
                    "source_split": ("source_split0",),
                },
                inval_key: {
                    "source_split": ("source_split0",),
                },
            },
            {
                CONDITION_LEVEL_NAME: {
                    "source_split": ("source_split0",),
                },
            },
        ],
    )
    def test_query_with_dict(
        self,
        adata: AnnData,
        groups_cols: Collection[str] | None,
        conditions_cols: Collection[str] | None,
        query_dict: dict[str, Any],
    ) -> None:
        # initialize indexer and selector
        indexer = HierarchicalIndexer(
            groups_cols=groups_cols,
            conditions_cols=conditions_cols,
        )
        selector = IndexSelector.init_from_indexer(indexer)
        index = indexer.create_index(adata.obs)

        # query index (fail cases)
        if inval_key in query_dict:
            with pytest.raises(ValueError):
                query_res = selector.query_with_dict(query_dict, index)
            return None
        elif CONDITION_LEVEL_NAME in query_dict and conditions_cols is None:
            with pytest.raises(ValueError):
                query_res = selector.query_with_dict(query_dict, index)
            return None
        elif CONDITION_LEVEL_NAME in query_dict and "source_split" in query_dict[CONDITION_LEVEL_NAME]:
            with pytest.raises(ValueError):
                query_res = selector.query_with_dict(query_dict, index)
            return None
        elif GROUP_LEVEL_NAME in query_dict and groups_cols is None:
            with pytest.raises(ValueError):
                query_res = selector.query_with_dict(query_dict, index)
            return None
        query_res = selector.query_with_dict(query_dict, index)
        assert isinstance(query_res, pd.MultiIndex)

    @pytest.mark.parametrize("groups_cols", [None, ["source_split"]])
    @pytest.mark.parametrize("conditions_cols", [None, ["drugA", "drugB"]])
    @pytest.mark.parametrize("level_name", [CONDITION_LEVEL_NAME, GROUP_LEVEL_NAME])
    def test_level_index_to_nested_dict(
        self,
        adata: AnnData,
        groups_cols: Collection[str] | None,
        conditions_cols: Collection[str] | None,
        level_name: str,
    ) -> None:
        # initialize indexer and selector
        indexer = HierarchicalIndexer(
            groups_cols=groups_cols,
            conditions_cols=conditions_cols,
        )
        selector = IndexSelector.init_from_indexer(indexer)
        index = indexer.create_index(adata.obs)

        # create nested dictionary (fail cases)
        nested_dict = selector.level_index_to_nested_dict(level_name, index)
        assert isinstance(nested_dict, NestedMappedLevelIndex)
