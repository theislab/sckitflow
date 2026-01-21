from anndata import AnnData

from sc_flow.data.containers._coupling import CouplingData, StateData
from sc_flow.data.schemas._base_schema import StrictDataSchema

__all__ = ["CouplingDataSchema"]


class CouplingDataSchema(StrictDataSchema):
    """"""  # noqa

    def __init__(
        self,
        source_rep: str | None = None,
        target_rep: str | None = None,
        n_shared_dims: int | None = None,
    ) -> None:
        """"""  # noqa
        self._source_rep = source_rep
        self._target_rep = target_rep
        self._n_shared_dims = n_shared_dims
        super().__init__()

    def _verify_args(self) -> None:
        """Verifies the validity of arguments set at initialization."""
        if self._n_shared_dims is not None:
            if self._n_shared_dims <= 0:
                msg = ""
                raise ValueError(msg)
            if self._target_rep == self._source_rep:
                msg = (
                    "When passing the number of shared dimension (i.e.: incomparable spaces), "
                    "you need to pass different representations for source and targets."
                )
                raise ValueError(msg)

    def _verify_schema(self, adata: AnnData) -> None:
        """Verifies the schema on the input :class: `AnnData`.

        :param adata: The input data.
        :type adata: class: `AnnData`
        """
        # verify target representation
        if self._target_rep is not None:
            self._check_key_found_in_adata_field(adata, self._target_rep, "obsm")

        # check rest that is ignored otherwise
        if self._n_shared_dims is not None:
            # verify source representation
            if self._source_rep is not None:
                self._check_key_found_in_adata_field(adata, self._source_rep, "obsm")
            # extracting data shape
            source_shape = self._extract_array(adata, repr=self._source_rep).shape
            target_shape = self._extract_array(adata, repr=self._target_rep).shape
            # verifying it with respect to the number of shared dimensions
            if self._n_shared_dims > source_shape[1]:
                msg = (
                    "Number of shared dims should be smaller than the number of source spatial dimensions. "
                    f"Found {source_shape[1]} spatial dimensions for the data, but {self._n_shared_dims} "
                    "shared dimensions were requested."
                )
                raise ValueError(msg)
            if self._n_shared_dims > target_shape[1]:
                msg = (
                    "Number of shared dims should be smaller than the number of target spatial dimensions. "
                    f"Found {target_shape[1]} spatial dimensions for the data, but {self._n_shared_dims} "
                    "shared dimensions were requested."
                )
                raise ValueError(msg)

    def _get_source_state_data(self, adata: AnnData) -> StateData:
        X = self._extract_array(adata, repr=self._source_rep)
        return StateData(X)

    def _get_target_state_data(self, adata: AnnData) -> StateData:
        """"""  # noqa
        X = self._extract_array(adata, repr=self._target_rep)
        return StateData(X)

    def _get_data(self, adata: AnnData) -> tuple[CouplingData, CouplingData]:
        """Enforces the schema on the input :class: `AnnData`.

        :param adata: The input data.
        :type adata: class: `AnnData`
        """
        target_state = self._get_target_state_data(adata)
        if self._n_shared_dims is None:
            source_coupling = CouplingData.init_from_state_data(target_state, n_shared_dims=self._n_shared_dims)
        else:
            source_state = self._get_source_state_data(adata)
            source_coupling = CouplingData.init_from_state_data(source_state, n_shared_dims=self._n_shared_dims)
        target_coupling = CouplingData.init_from_state_data(target_state, n_shared_dims=self._n_shared_dims)
        return source_coupling, target_coupling

    @property
    def source_rep(self) -> str | None:
        """"""  # noqa
        return self._source_rep

    @property
    def target_rep(self) -> str | None:
        """"""  # noqa
        return self._target_rep

    @property
    def n_shared_dims(self) -> int | None:
        """"""  # noqa
        return self._n_shared_dims

    @property
    def has_incomparable_spaces(self) -> bool:
        """"""  # noqa
        return self._n_shared_dims is not None
