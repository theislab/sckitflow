from collections.abc import Collection
from typing import Any

import pandas as pd
import pytest
from anndata import AnnData

from sc_flow._constants import CONDITION_LEVEL_NAME, GROUP_LEVEL_NAME
from sc_flow.data.grouping._indexer import HierarchicalIndexer
from sc_flow.data.grouping._selector import IndexSelector

wrong_key = "wrong_key"


class TestIndexSelector:
    @pytest.mark.parametrize("groups_cols", [None, ["source_split"]])
    @pytest.mark.parametrize("conditions_cols", [None, ["drugA", "drugB"]])
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

        # creating index
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
