from collections.abc import Collection
from typing import get_args

import pytest
from anndata import AnnData

from sc_flow._types import TargetCovariatesEncodingId
from sc_flow.data._structures import CategoricalData
from sc_flow.data.schemas import GroupsDataSchema

inval_key: str = "invalid_key"


class TestGroupsDataSchema:
    def _is_valid_args(
        self,
        groups: Collection[str] | None,
        groups_reps: dict[str, str] | None,
        groups_encoding: dict[str, TargetCovariatesEncodingId] | None,
    ) -> bool:
        if groups is None and (groups_reps is not None or groups_encoding is not None):
            return False
        elif groups is not None and (groups_reps is None and groups_encoding is None):
            return False
        elif groups is not None and groups_reps is not None and groups_encoding is not None:
            shared_keys = set(groups_reps.keys()).intersection(set(groups_encoding.keys()))
            all_keys = set(groups_reps.keys()).union(set(groups_encoding.keys()))
            if len(shared_keys) > 0:
                return False
        elif groups is not None and groups_reps is not None and groups_encoding is None:
            all_keys = set(groups_reps.keys())
            if all_keys != set(groups):
                return False
        elif groups is not None and groups_reps is None and groups_encoding is not None:
            all_keys = set(groups_encoding.keys())
            if all_keys != set(groups):
                return False
        if groups_encoding is not None:
            for v in groups_encoding.values():
                if v not in get_args(TargetCovariatesEncodingId):
                    return False
        return True

    @pytest.mark.parametrize(
        "groups",
        [
            None,
            ["source_split"],
        ],
    )
    @pytest.mark.parametrize(
        "groups_reps",
        [None, {"source_split": "source_split"}, {inval_key: "source_split"}],
    )
    @pytest.mark.parametrize(
        "groups_encoding",
        [None, {"source_split": "source_split"}, {inval_key: "source_split"}],
    )
    def test_init(
        self,
        groups: Collection[str] | None,
        groups_reps: dict[str, str] | None,
        groups_encoding: dict[str, TargetCovariatesEncodingId] | None,
    ) -> None:
        is_valid_args = self._is_valid_args(
            groups,
            groups_reps,
            groups_encoding,
        )
        if not is_valid_args:
            with pytest.raises(ValueError):
                _schema = GroupsDataSchema(
                    groups=groups,
                    groups_reps=groups_reps,
                    groups_encoding=groups_encoding,
                )
            return None
        _schema = GroupsDataSchema(
            groups=groups,
            groups_reps=groups_reps,
            groups_encoding=groups_encoding,
        )

    @pytest.mark.parametrize(
        "groups",
        [
            ["source_split"],
            [inval_key],
        ],
    )
    @pytest.mark.parametrize(
        "groups_reps",
        [None, {"source_split": "source_split"}, {"source_split": inval_key}],
    )
    @pytest.mark.parametrize(
        "groups_encoding",
        [
            None,
            {"source_split": "one-hot"},
        ],
    )
    def test_get_data(
        self,
        adata: AnnData,
        groups: Collection[str] | None,
        groups_reps: dict[str, str] | None,
        groups_encoding: dict[str, TargetCovariatesEncodingId] | None,
    ) -> None:
        is_valid_args = self._is_valid_args(
            groups,
            groups_reps,
            groups_encoding,
        )
        if not is_valid_args:
            pytest.skip()
        schema = GroupsDataSchema(
            groups=groups,
            groups_reps=groups_reps,
            groups_encoding=groups_encoding,
        )
        if inval_key in groups or (groups_reps is not None and inval_key in groups_reps.values()):
            with pytest.raises(KeyError):
                data = schema.get_data(adata)
            return None
        data = schema.get_data(adata)
        assert isinstance(data, CategoricalData)
