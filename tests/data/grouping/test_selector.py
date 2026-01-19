from collections.abc import Collection
from typing import Any

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from sc_flow._constants import BASE_LEVEL_NAME, CONDITION_LEVEL_NAME, GROUP_LEVEL_NAME
from sc_flow._types import MappedLevelIndex
from sc_flow.data.grouping._indexer import HierarchicalIndexer
from sc_flow.data.grouping._selector import IndexSelector

inval_key = "wrong_key"


class TestIndexSelector:
    def _validate_correct_unique_values(
        self,
        df: pd.DataFrame,
        nested_dict: MappedLevelIndex,
        cols: Collection[str] | None,
    ) -> None:
        """"""
        if cols is not None and len(cols) > 0:
            unique_groups = df.loc[:, cols].drop_duplicates().values
            unique_groups = map(tuple, unique_groups)
        else:
            unique_groups = [()] * len(df)
        assert set(nested_dict.mapping.keys()) == set(unique_groups)

    def _extract_df(self, reference_df, cols, key):
        """Subset the dataframe corresponding to this group key."""
        if not cols:
            return reference_df, []
        mask = (reference_df.loc[:, cols].values == np.asarray(key)).all(-1)
        df = reference_df.loc[mask]
        values = list(map(tuple, df.loc[:, cols].values))
        return df, values

    def _build_expected_index(self, cond_df, group_vals, cond_vals):
        """Create the expected MultiIndex for comparison."""
        tuples = []
        groups_vals = group_vals if len(group_vals) > 0 else [()] * len(cond_df)
        cond_vals = cond_vals if len(cond_vals) > 0 else [()] * len(cond_df)
        for idx, gvals, cvals in zip(cond_df.index, groups_vals, cond_vals, strict=False):
            tuples.append((idx, gvals, cvals))
        names = [BASE_LEVEL_NAME, GROUP_LEVEL_NAME, CONDITION_LEVEL_NAME]
        return pd.MultiIndex.from_tuples(tuples, names=names)

    def _validate_nested_dict(
        self,
        reference_df: pd.DataFrame,
        nested_dict: MappedLevelIndex,
        groups_cols: Collection[str] | None,
        conditions_cols: Collection[str] | None,
    ) -> None:
        # Validate top-level keys
        self._validate_correct_unique_values(reference_df, nested_dict, groups_cols)

        # Iterate group keys
        for group_key, group_node in nested_dict.mapping.items():
            assert isinstance(group_node, MappedLevelIndex)

            # Group-level subset
            group_df, group_values = self._extract_df(reference_df, groups_cols, group_key)

            # Validate second-level keys
            self._validate_correct_unique_values(group_df, group_node, conditions_cols)

            # Iterate condition keys
            for cond_key, cond_index in group_node.mapping.items():
                cond_df, cond_values = self._extract_df(group_df, conditions_cols, cond_key)
                expected_index = self._build_expected_index(
                    cond_df,
                    group_values,
                    cond_values,
                )

                # Final checks
                assert isinstance(cond_index, pd.MultiIndex)
                assert set(cond_index) == set(expected_index)

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
    def test_index_to_nested_dict(
        self,
        adata: AnnData,
        groups_cols: Collection[str] | None,
        conditions_cols: Collection[str] | None,
    ) -> None:
        # initialize indexer and selector
        indexer = HierarchicalIndexer(
            groups_cols=groups_cols,
            conditions_cols=conditions_cols,
        )
        selector = IndexSelector.init_from_indexer(indexer)
        index = indexer.create_index(adata.obs)

        # create nested dictionary (fail cases)
        nested_dict = selector.index_to_nested_dict(index)
        assert isinstance(nested_dict, MappedLevelIndex)
        self._validate_nested_dict(
            adata.obs,
            nested_dict,
            groups_cols,
            conditions_cols,
        )
