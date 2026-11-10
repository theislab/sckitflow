from sckitflow.data._utils import Rep
from sckitflow.data.schemas._base_schema import StrictDataSchema

__all__ = ["CouplingDataSchema"]


class CouplingDataSchema(StrictDataSchema):
    """Data schema implementing the logic for coupling data.

    Coupling data is the observation representation used to match source and target
    distributions. Nothing is read unless a rep is named; three modes are supported:

    * **Linear only** — ``rep_lin`` set, no ``*_rep_quad`` configured.
      Source and target are matched in a shared space.
    * **Fused** — ``rep_lin`` with ``source_rep_quad`` / ``target_rep_quad``.
      Shared axes plus per-side geometry (fused Gromov-Wasserstein).
    * **Pure Gromov-Wasserstein** — ``use_linear=False`` with both quadratics set.
      Source and target are matched on intra-domain geometry alone.
    """

    def __init__(
        self,
        rep_lin: Rep | None = None,
        source_rep_quad: Rep | None = None,
        target_rep_quad: Rep | None = None,
        use_linear: bool = True,
    ) -> None:
        """Initializes the coupling data schema.

        :param rep_lin: Optional :mod:`anndata.acc` reference (``A.X``, ``A.layers[k]`` or
            ``A.obsm[k]``) to the linear representation shared by source and target
            distribution samples, the shared axes between the source and the target spaces.
            Required for a linear term, as there is no default location (pass ``A.X`` for
            `.X`). Defaults to `None`: no linear term. Must be `None` when `use_linear=False`.
        :type rep_lin: class: `Rep | None`

        :param source_rep_quad: Optional :mod:`anndata.acc` reference to the representation
            for the quadratic term of the source distribution samples. It holds
            the features defined only on the source space.
        :type source_rep_quad: class: `Rep | None`

        :param target_rep_quad: Optional :mod:`anndata.acc` reference to the representation
            for the quadratic term of the target distribution samples. It holds
            the features defined only on the target space.
        :type target_rep_quad: class: `Rep | None`

        :param use_linear: Whether the coupling has a shared linear term. Set to `False`
            for pure Gromov-Wasserstein, where source and target are matched on
            intra-domain geometry alone and share no axes. Requires both
            `source_rep_quad` and `target_rep_quad`, and is mutually exclusive with
            `rep_lin`. Defaults to `True`.
        :type use_linear: class: `bool`
        """
        if use_linear and rep_lin is None and (source_rep_quad is not None or target_rep_quad is not None):
            raise ValueError(
                "a fused coupling (quadratic terms with a shared linear term) needs `rep_lin`, the rep source "
                "and target share (e.g. 'X' or 'obsm.X_pca'); there is no default. For pure Gromov-Wasserstein, "
                "with no linear term, set `use_linear=False`."
            )
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

    @property
    def rep_lin(self) -> Rep | None:
        """Exposes the `rep_lin` parameter set at initialization."""
        return self._rep_lin

    @property
    def source_rep_quad(self) -> Rep | None:
        """Exposes the `source_rep_quad` parameter set at initialization."""
        return self._source_rep_quad

    @property
    def target_rep_quad(self) -> Rep | None:
        """Exposes the `target_rep_quad` parameter set at initialization."""
        return self._target_rep_quad

    @property
    def use_linear(self) -> bool:
        """Whether a shared linear term is part of the coupling."""
        return self._use_linear
