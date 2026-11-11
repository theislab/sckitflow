from sckitflow.data._utils import Rep
from sckitflow.data.schemas._base_schema import StrictDataSchema

__all__ = ["CouplingDataSchema"]


class CouplingDataSchema(StrictDataSchema):
    """Data schema implementing the logic for coupling data.

    Coupling data is the observation representation used to match source and target
    distributions. Nothing is read unless a rep is named; three modes are supported:

    * **Linear only**: ``rep_lin`` set, no ``*_rep_quad``.
      Source and target are matched in a shared space.
    * **Fused**: ``rep_lin`` with ``source_rep_quad`` and ``target_rep_quad``.
      Shared axes plus per-side geometry (fused Gromov-Wasserstein).
    * **Pure Gromov-Wasserstein**: both quadratics, no ``rep_lin``.
      Source and target are matched on intra-domain geometry alone.

    One rep may serve several terms, e.g. ``rep_lin`` and ``source_rep_quad``.
    """

    def __init__(
        self,
        rep_lin: Rep | None = None,
        source_rep_quad: Rep | None = None,
        target_rep_quad: Rep | None = None,
    ) -> None:
        """Initializes the coupling data schema.

        :param rep_lin: Optional :mod:`anndata.acc` reference (``A.X``, ``A.layers[k]`` or ``A.obsm[k]``) to the
            representation source and target share. ``None``: no linear term.
        :type rep_lin: class: `Rep | None`

        :param source_rep_quad: Optional reference to the sources' representation for the quadratic term, the
            features defined only on the source space.
        :type source_rep_quad: class: `Rep | None`

        :param target_rep_quad: Optional reference to the targets' representation for the quadratic term, the
            features defined only on the target space.
        :type target_rep_quad: class: `Rep | None`
        """
        if rep_lin is None and (source_rep_quad is None) != (target_rep_quad is None):
            raise ValueError(
                "without `rep_lin` (pure Gromov-Wasserstein) both `source_rep_quad` and `target_rep_quad` are needed; "
                "with one missing, its side has nothing to match over."
            )
        self._rep_lin = rep_lin
        self._source_rep_quad = source_rep_quad
        self._target_rep_quad = target_rep_quad
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
    def source_terms(self) -> dict[str, Rep]:
        """The sources' coupling terms that are set, ``{StepData key: reference}``."""
        terms = {"source_coupling_lin": self._rep_lin, "source_coupling_quad": self._source_rep_quad}
        return {t: r for t, r in terms.items() if r is not None}

    @property
    def target_terms(self) -> dict[str, Rep]:
        """The targets' coupling terms that are set, ``{StepData key: reference}``."""
        terms = {"target_coupling_lin": self._rep_lin, "target_coupling_quad": self._target_rep_quad}
        return {t: r for t, r in terms.items() if r is not None}
