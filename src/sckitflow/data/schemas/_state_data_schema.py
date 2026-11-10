from sckitflow.data._utils import Rep
from sckitflow.data.schemas._base_schema import StrictDataSchema

__all__ = ["StateDataSchema"]


class StateDataSchema(StrictDataSchema):
    """Data Schema Implementing the logic for state data."""

    def __init__(
        self,
        sample_rep: Rep,
    ) -> None:
        """Initializes the state data schema

        :param sample_rep: The representation to extract, as an :mod:`anndata.acc` reference (``A.X``,
            ``A.layers[k]`` or ``A.obsm[k]``).
        :type sample_rep: `Rep`
        """
        self._sample_rep = sample_rep
        super().__init__()

    def _verify_args(self) -> None:
        """No-op needed for compatibility with parent class."""
        pass

    @property
    def sample_rep(self) -> Rep:
        """Exposes to `sample_rep` parameter set at initialization."""
        return self._sample_rep
