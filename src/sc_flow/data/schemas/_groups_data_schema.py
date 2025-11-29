from collections.abc import Collection
from dataclasses import dataclass
from dataclasses import field as dc_field

from anndata import AnnData

from sc_flow._types import TargetCovariatesEncodingId
from sc_flow.data._structures import GroupsData
from sc_flow.data.schemas._base_schema import BaseDataSchema


@dataclass
class GroupsDataSchema(BaseDataSchema):
    """"""  # noqa

    groups: Collection[str] = dc_field(default_factory=lambda: [])
    groups_reps: dict[str, str] = dc_field(default_factory=lambda: {})
    groups_encoding: dict[str, TargetCovariatesEncodingId] = dc_field(default_factory=lambda: {})

    def _verify_schema(self, adata: AnnData) -> None:
        """"""  # noqa
        raise NotImplementedError

    def _get_data(self, adata: AnnData) -> GroupsData:
        """"""  # noqa
        raise NotImplementedError
