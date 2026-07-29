from collections.abc import Collection

import pytest
from anndata import AnnData

from sc_flow.data._group_encoders import GroupEncoder, OneHot
from sc_flow.data.containers import CategoricalData
from sc_flow.data.schemas import GroupsDataSchema

from ..shared import verify_categorical_data  # noqa

inval_key: str = "invalid_key"


class TestGroupsDataSchema:
    def _is_valid_args(
        self,
        groups: Collection[str] | None,
        groups_reps: dict[str, str] | None,
        groups_encoding: dict[str, GroupEncoder] | None,
    ) -> bool:
        # The schema requires reps/encoding keys to partition exactly the group columns.
        reps_keys = set() if groups_reps is None else set(groups_reps)
        enc_keys = set() if groups_encoding is None else set(groups_encoding)
        if groups is None:
            return not reps_keys and not enc_keys
        if reps_keys & enc_keys:
            return False
        return reps_keys | enc_keys == set(groups)

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
        [None, {"source_split": OneHot()}, {inval_key: OneHot()}],
    )
    def test_init(
        self,
        groups: Collection[str] | None,
        groups_reps: dict[str, str] | None,
        groups_encoding: dict[str, GroupEncoder] | None,
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
            {"source_split": OneHot()},
        ],
    )
    def test_get_data(
        self,
        adata: AnnData,
        uns_keys_to_nunique_prefix_and_dim: dict[str, tuple[int, str, int]],
        groups: Collection[str] | None,
        groups_reps: dict[str, str] | None,
        groups_encoding: dict[str, GroupEncoder] | None,
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
        verify_categorical_data(
            data,
            groups,
            uns_keys_to_nunique_prefix_and_dim,
            conditions_reps=groups_reps,
            groups_encoding=groups_encoding,
        )
