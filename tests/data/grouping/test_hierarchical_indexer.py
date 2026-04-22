from collections.abc import Collection

import pandas as pd
import pytest
from anndata import AnnData

from sc_flow.data._utils import convert_to_categorical_in_place
from sc_flow.data.grouping._indexer import HierarchicalIndexer

wrong_key = "wrong_key"


class TestHierarchicalIndexer:
    @staticmethod
    def _verify_level_index(
        level_name: str,
        level_cols: Collection[str],
        idxs: pd.MultiIndex,
    ) -> None:
        sub_level_names = [n for n in idxs.names if isinstance(n, tuple) and n[0] == level_name]
        if level_cols is None or len(level_cols) == 0:
            assert len(sub_level_names) == 0
        else:
            assert len(sub_level_names) == len(level_cols)

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

        obs = adata.obs.copy()
        convert_to_categorical_in_place(obs, groups_cols)
        convert_to_categorical_in_place(obs, conditions_cols)

        if groups_cols is not None and wrong_key in groups_cols:
            with pytest.raises(ValueError, match=r"The following query columns dont appear in the reference:"):
                idxs = indexer.create_index(obs)
            return None
        if conditions_cols is not None and wrong_key in conditions_cols:
            with pytest.raises(ValueError, match=r"The following query columns dont appear in the reference:"):
                idxs = indexer.create_index(obs)
            return None
        idxs = indexer.create_index(obs)

        self._verify_level_index("groups", groups_cols, idxs)
        self._verify_level_index("conditions", conditions_cols, idxs)

    @pytest.mark.parametrize("level_name", ["groups", "knockout"])
    @pytest.mark.parametrize(
        "level_cols",
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
        level_cols: Collection[str] | None,
        allow_override: bool,
    ) -> None:
        indexer = HierarchicalIndexer(
            groups_cols=["source_split"],
            conditions_cols=["drugA", "drugB"],
        )

        if not allow_override and level_name == "groups":
            with pytest.raises(ValueError, match=r"Level .* already present, cannot override\."):
                indexer.update_registry(level_name, level_cols=level_cols, allow_override=allow_override)
            return None
        indexer.update_registry(level_name, level_cols=level_cols, allow_override=allow_override)
        if level_name == "groups":
            assert len(indexer.registry) == 2, indexer.registry.keys()
        else:
            assert len(indexer.registry) == 3, indexer.registry.keys()

        all_cols = ["source_split", "drugA", "drugB"]
        if level_cols is not None:
            all_cols.extend(level_cols)
        obs = adata.obs.copy()
        convert_to_categorical_in_place(obs, all_cols)

        if level_cols is not None and wrong_key in level_cols:
            with pytest.raises(ValueError, match=r"The following query columns dont appear in the reference:"):
                idxs = indexer.create_index(obs)
            return None
        idxs = indexer.create_index(obs)

        self._verify_level_index(level_name, level_cols, idxs)
