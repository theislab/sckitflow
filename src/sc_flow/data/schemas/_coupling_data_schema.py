from anndata import AnnData

from sc_flow.data._structures import CouplingData, StateData
from sc_flow.data.schemas._base_schema import BaseDataSchema

__all__ = ["CouplingDataSchema"]


class CouplingDataSchema(BaseDataSchema):
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
            source_shape = (adata.X if self._source_rep is None else adata.obsm[self._source_rep]).shape
            target_shape = (adata.X if self._target_rep is None else adata.obsm[self._target_rep]).shape
            # verifying it with respect to the number of shared dimensions
            if self._n_shared_dims >= source_shape[1] or self._n_shared_dims >= target_shape[1]:
                msg = ""
                raise ValueError(msg)

    def _get_target_lin_data(self, adata: AnnData) -> StateData:
        """"""  # noqa
        X = self._extract_array(adata, rep=self._target_rep)
        if self.has_incomparable_spaces:
            X = X[:, : self._n_shared_dims]
        return StateData(X)

    def _get_target_quad_data(self, adata: AnnData) -> StateData | None:
        """"""  # noqa
        # early return when spaces are comparable
        if not self.has_incomparable_spaces:
            return None
        X = self._extract_array(adata, rep=self._target_rep)
        X = X[:, self._n_shared_dims :]
        return StateData(X)

    def _get_source_lin_data(self, adata: AnnData) -> StateData | None:
        """"""  # noqa
        # early return when spaces are comparable
        if not self.has_incomparable_spaces:
            return None
        X = self._extract_array(adata, rep=self._source_rep)
        X = X[:, : self._n_shared_dims]
        return StateData(X)

    def _get_source_quad_data(self, adata: AnnData) -> StateData | None:
        """"""  # noqa
        # early return when spaces are comparable
        if not self.has_incomparable_spaces:
            return None
        X = self._extract_array(adata, rep=self._source_rep)
        X = X[:, self._n_shared_dims :]
        return StateData(X)

    def _get_data(self, adata: AnnData) -> tuple[CouplingData | None, CouplingData]:
        """Enforces the schema on the input :class: `AnnData`.

        :param adata: The input data.
        :type adata: class: `AnnData`
        """
        target_lin: StateData = self._get_target_lin_data(adata)
        target_quad: StateData | None = self._get_target_quad_data(adata)
        source_lin: StateData | None = self._get_source_lin_data(adata)
        source_quad: StateData | None = self._get_source_quad_data(adata)
        target_coupling = CouplingData(target_lin, state_quad=target_quad)
        if source_lin is not None:
            source_coupling = CouplingData(source_lin, state_quad=source_quad)
        else:
            source_coupling = None
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
