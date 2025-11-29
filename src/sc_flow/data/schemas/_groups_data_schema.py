from collections.abc import Collection
from dataclasses import dataclass

from anndata import AnnData

from sc_flow._types import TargetCovariatesEncodingId
from sc_flow.data._structures import GroupsData
from sc_flow.data.schemas._base_schema import BaseDataSchema


@dataclass
class GroupsDataSchema(BaseDataSchema):
    """"""  # noqa

    groups: dict[str, Collection[str]] | None = None
    groups_reps: dict[str, str] | None = None
    groups_encoding: dict[str, TargetCovariatesEncodingId] | None = None

    def _verify_schema(self, adata: AnnData) -> None:
        """"""  # noqa
        raise NotImplementedError

    def _get_data(self, adata: AnnData) -> GroupsData:
        """"""  # noqa
        raise NotImplementedError
