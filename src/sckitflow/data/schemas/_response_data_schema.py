from collections.abc import Collection

from sckitflow.data._utils import Rep
from sckitflow.data.schemas._base_schema import StrictDataSchema

__all__ = ["ResponseDataSchema"]


class ResponseDataSchema(StrictDataSchema):
    """Data Schema implementing the logic for response data."""

    def __init__(self, continuous_covs: Collection[Rep] | None = None) -> None:
        """Initializes the data schema.

        :param continuous_covs: Sequence of continuous response covariates, as :mod:`anndata.acc` references
            (``A.obsm[k]`` or ``A.layers[k]``). Defaults to `None`.
        :type continuous_covs: class: `Collection[Rep] | None`
        """
        self._continuous_covs = [] if continuous_covs is None else continuous_covs
        super().__init__()

    def _verify_args(self) -> None:
        """Nothing to verify: the keys are checked when the dimensions are read."""

    @property
    def continuous_covs(self) -> Collection[Rep]:
        """Exposes to `continuous_covs` parameter set at initialization."""
        return self._continuous_covs

    @property
    def has_continuous_covariates(self) -> bool:
        """Whether the response schema includes continuous covariates."""
        return len(self._continuous_covs) > 0
