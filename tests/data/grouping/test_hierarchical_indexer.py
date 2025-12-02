from collections.abc import Collection

import pandas as pd
import pytest
from anndata import AnnData

from sc_flow.data.grouping._indexer import HierarchicalIndexer

wrong_key = "wrong_key"


class TestHierarchicalIndexer:
    @staticmethod
    def verify_level_index(
        level_name: str,
        level_cols: Collection[str],
        idxs: pd.MultiIndex,
    ) -> None:
        level_values = idxs.get_level_values(level_name)
        if level_cols is None:
            dummy_val = f"{level_name}_dummy"
            assert set(level_values) == {((dummy_val,))}
        else:
            assert len(level_cols) == len(level_values[0])

    @pytest.mark.parametrize("groups_cols", [None, ["source_split"], [wrong_key]])
    @pytest.mark.parametrize("conditions_cols", [None, ["drugA", "drugB"], [wrong_key]])
    def test_create_index(
        self, adata: AnnData, groups_cols: Collection[str] | None, conditions_cols: Collection[str] | None
    ) -> None:
        indexer = HierarchicalIndexer(
            groups_cols=groups_cols,
            conditions_cols=conditions_cols,
        )
        assert len(indexer.registry) == 2, indexer.registry.keys()

        if groups_cols is not None and wrong_key in groups_cols:
            with pytest.raises(KeyError, match=r"Columns .* not in dataframe\."):
                idxs = indexer.create_index(adata.obs)
            return None
        if conditions_cols is not None and wrong_key in conditions_cols:
            with pytest.raises(KeyError, match=r"Columns .* not in dataframe\."):
                idxs = indexer.create_index(adata.obs)
            return None
        idxs = indexer.create_index(adata.obs)

        self.verify_level_index("groups", groups_cols, idxs)
        self.verify_level_index("conditions", conditions_cols, idxs)

    @pytest.mark.parametrize("level_name", ["groups", "knockout"])
    @pytest.mark.parametrize(
        "level_columns",
        [
            None,
            ["koA", "koB"],
            [wrong_key],
        ],
    )
    @pytest.mark.parametrize("allow_override", [True, False])
    def test_update_registry(
        self,
        adata: AnnData,
        level_name: str,
        level_columns: Collection[str] | None,
        allow_override: bool,
    ) -> None:
        indexer = HierarchicalIndexer(
            groups_cols=["source_split"],
            conditions_cols=["drugA", "drugB"],
        )

        if not allow_override and level_name == "groups":
            with pytest.raises(ValueError, match=r"Level .* already present, cannot override\."):
                indexer.update_registry(level_name, level_columns=level_columns, allow_override=allow_override)
            return None
        indexer.update_registry(level_name, level_columns=level_columns, allow_override=allow_override)
        if level_name == "groups":
            assert len(indexer.registry) == 2, indexer.registry.keys()
        else:
            assert len(indexer.registry) == 3, indexer.registry.keys()

        if level_columns is not None and wrong_key in level_columns:
            with pytest.raises(KeyError, match=r"Columns .* not in dataframe\."):
                idxs = indexer.create_index(adata.obs)
            return None
        idxs = indexer.create_index(adata.obs)

        self.verify_level_index(level_name, level_columns, idxs)
