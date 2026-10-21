from anndata import AnnData

from sckitflow.data.containers._coupling import CouplingData, StateData
from sckitflow.data.schemas._base_schema import StrictDataSchema

__all__ = ["CouplingDataSchema"]


class CouplingDataSchema(StrictDataSchema):
    """Data schema implementing the logic for coupling data.

    Coupling data is the observation representation used to match source and target
    distributions. Three modes are supported:

    * **Linear only** — ``use_linear=True`` (default), no ``*_rep_quad`` configured.
      Source and target are matched in a shared space.
    * **Fused** — ``use_linear=True`` with ``source_rep_quad`` / ``target_rep_quad``.
      Shared axes plus per-side geometry (fused Gromov-Wasserstein).
    * **Pure Gromov-Wasserstein** — ``use_linear=False`` with both quadratics set.
      Source and target are matched on intra-domain geometry alone.
    """

    def __init__(
        self,
        rep_lin: str | None = None,
        source_rep_quad: str | None = None,
        target_rep_quad: str | None = None,
        use_linear: bool = True,
    ) -> None:
        """Initializes the coupling data schema.

        :param rep_lin: Optional string identifier denoting the linear representation
            shared by source and target distribution samples. It should be a key
            appearing in the `.obsm` of annotated data objects. Defaults to `None`,
            in which case `.X` is used. This location identifies the shared axes between
            the source and the target spaces. Ignored (and must be `None`) when
            `use_linear=False`.
        :type rep_lin: class: `str | None`

        :param source_rep_quad: Optional string identifier denoting the representation
            for the quadratic term of the source distribution samples. It should be a
            key appearing in the `.obsm` of annotated data objects. This loc contains
            the features defined only on the source space.
        :type source_rep_quad: class: `str | None`

        :param target_rep_quad: Optional string identifier denoting the representation
            for the quadratic term of the target distribution samples. It should be a
            key appearing in the `.obsm` of annotated data objects. This loc contains
            the features defined only on the target space.
        :type target_rep_quad: class: `str | None`

        :param use_linear: Whether the coupling has a shared linear term. Set to `False`
            for pure Gromov-Wasserstein, where source and target are matched on
            intra-domain geometry alone and share no axes. Requires both
            `source_rep_quad` and `target_rep_quad`, and is mutually exclusive with
            `rep_lin`. Defaults to `True`.
        :type use_linear: class: `bool`
        """
        if not use_linear:
            if rep_lin is not None:
                raise ValueError(
                    "`rep_lin` is redundant when `use_linear=False`: there is no shared "
                    "linear term to read. Pass `use_linear=False` alone, or drop it and "
                    "set `rep_lin`."
                )
            if source_rep_quad is None or target_rep_quad is None:
                raise ValueError(
                    "`use_linear=False` (pure Gromov-Wasserstein) requires both "
                    "`source_rep_quad` and `target_rep_quad`; without a linear term and "
                    "with a missing quadratic, one side would have nothing to match over."
                )

        self._rep_lin = rep_lin
        self._source_rep_quad = source_rep_quad
        self._target_rep_quad = target_rep_quad
        self._use_linear = use_linear
        super().__init__()

    def _verify_args(self) -> None:
        """No-op needed for compatibility with parent class."""
        pass

    def _verify_schema(self, adata: AnnData) -> None:
        """Verifies the schema on the input :class: `AnnData`.

        :param adata: The input data.
        :type adata: class: `AnnData`
        """
        if self._use_linear and self._rep_lin is not None:
            self._check_key_found_in_adata_field(adata, self._rep_lin, "obsm")
        if self._source_rep_quad is not None:
            self._check_key_found_in_adata_field(adata, self._source_rep_quad, "obsm")
        if self._target_rep_quad is not None:
            self._check_key_found_in_adata_field(adata, self._target_rep_quad, "obsm")

    def get_sides(self, adata: AnnData, *, source: bool, target: bool) -> tuple[CouplingData, CouplingData]:
        """Like :meth:`get_data`, but reads a side's quadratic rep only when ``adata`` holds that side's cells.

        A dataset split across files keeps each side's quadratic rep where its cells are; the linear term is
        read from every file, since both sides compare in it.
        """
        reads = {
            "source": self._source_rep_quad if source else None,
            "target": self._target_rep_quad if target else None,
        }
        if self._use_linear and self._rep_lin is not None:
            self._check_key_found_in_adata_field(adata, self._rep_lin, "obsm")
        for key in reads.values():
            if key is not None:
                self._check_key_found_in_adata_field(adata, key, "obsm")
        return self._read(adata, reads["source"], reads["target"])

    def _get_data(self, adata: AnnData) -> tuple[CouplingData, CouplingData]:
        """Enforces the schema on the input :class: `AnnData`.

        :param adata: The input data.
        :type adata: class: `AnnData`
        """
        return self._read(adata, self._source_rep_quad, self._target_rep_quad)

    def _read(
        self, adata: AnnData, source_rep_quad: str | None, target_rep_quad: str | None
    ) -> tuple[CouplingData | None, CouplingData | None]:
        # Linear term (shared): `None` under pure Gromov-Wasserstein.
        if self._use_linear:
            X_lin = StateData(self._extract_array(adata, repr=self._rep_lin))
        else:
            X_lin = None

        # Quadratic terms, collected separately (specific dimensions).
        if source_rep_quad is not None:
            X_source_quad = StateData(self._extract_array(adata, repr=source_rep_quad))
        else:
            X_source_quad = None
        if target_rep_quad is not None:
            X_target_quad = StateData(self._extract_array(adata, repr=target_rep_quad))
        else:
            X_target_quad = None

        # A side with no term here (pure Gromov-Wasserstein, in a file not holding that side) has no coupling data.
        def side(quad: StateData | None) -> CouplingData | None:
            return None if X_lin is None and quad is None else CouplingData(state_lin=X_lin, state_quad=quad)

        return side(X_source_quad), side(X_target_quad)

    @property
    def rep_lin(self) -> str | None:
        """Exposes the `rep_lin` parameter set at initialization."""
        return self._rep_lin

    @property
    def source_rep_quad(self) -> str | None:
        """Exposes the `source_rep_quad` parameter set at initialization."""
        return self._source_rep_quad

    @property
    def target_rep_quad(self) -> str | None:
        """Exposes the `target_rep_quad` parameter set at initialization."""
        return self._target_rep_quad

    @property
    def use_linear(self) -> bool:
        """Whether a shared linear term is part of the coupling."""
        return self._use_linear
