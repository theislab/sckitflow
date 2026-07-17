from collections.abc import Mapping

from anndata import AnnData
from anndata.acc import A

from sc_flow.data.schemas._base_schema import StrictDataSchema

__all__ = ["CouplingDataSchema"]

# A coupling reference: an `anndata.acc` accessor (e.g. ``A.obsm["X_pca"]``) or a str resolved to one
# (a bare ``.obsm`` key, or ``"X"``). Materialized with ``adata[ref]``; presence-checked with
# ``ref in adata`` — no ``.obsm``-vs-``.X`` branching.
Ref = object


class CouplingDataSchema(StrictDataSchema):
    """Data schema for OT coupling — role-named source/target references (schema-generalization Change 3).

    Coupling is "match ``srcs`` to ``tgts``". Instead of ``source_rep``/``target_rep``/``n_shared_dims``,
    each side names its **linear** and/or **quadratic** space explicitly: ``src_lin``/``src_quad`` for the
    source (control) cells, ``tgt_lin``/``tgt_quad`` for the target (perturbed) cells. Each is an
    ``anndata.acc`` accessor (``A.obsm["X_pca"]``) or a str resolved to one (a bare ``.obsm`` key, or
    ``"X"``); no slicing, the reps arrive pre-separated.

    The regime is **inferred, not declared**: if ``*_quad`` are given the coupling is quadratic/GW, else
    linear (:attr:`is_quadratic`). ``*_lin`` and ``*_quad`` must be given symmetrically (both sides), and
    at least one regime must be present.

    There is no ``CouplingData`` container — coupling is per-minibatch. ``compile_obs`` streams these reps
    as extra, aligned binded ``Node`` keys (only when a coupling space differs from the state rep), so
    they arrive in the batch's ``source_reps`` / ``target_reps``; :attr:`refs` records role → location.
    """

    def __init__(
        self,
        src_lin: Ref | None = None,
        src_quad: Ref | None = None,
        tgt_lin: Ref | None = None,
        tgt_quad: Ref | None = None,
    ) -> None:
        self._raw: dict[str, Ref | None] = {
            "src_lin": src_lin,
            "src_quad": src_quad,
            "tgt_lin": tgt_lin,
            "tgt_quad": tgt_quad,
        }
        super().__init__()

    @staticmethod
    def _as_ref(ref: Ref | None):
        """Normalize a str (bare ``.obsm`` key, or ``"X"``) to an ``anndata.acc`` accessor; pass accessors through."""
        if ref is None:
            return None
        if isinstance(ref, str):
            return A.X if ref == "X" else A.obsm[ref]
        return ref

    def _verify_args(self) -> None:
        """Regimes must be symmetric across sides, and at least one regime must be present."""
        r = self._raw
        if (r["src_lin"] is None) != (r["tgt_lin"] is None):
            raise ValueError("linear coupling needs both src_lin and tgt_lin (or neither).")
        if (r["src_quad"] is None) != (r["tgt_quad"] is None):
            raise ValueError("quadratic coupling needs both src_quad and tgt_quad (or neither).")
        if all(v is None for v in r.values()):
            raise ValueError("CouplingDataSchema needs at least one regime (lin and/or quad).")

    def _verify_schema(self, adata: AnnData) -> None:
        """Every provided reference must resolve in ``adata`` (``ref in adata``)."""
        for role, ref in self.refs.items():
            if ref not in adata:
                raise KeyError(f"coupling {role} reference not found in adata.")

    @property
    def refs(self) -> Mapping[str, object]:
        """Present roles → their resolved ``anndata.acc`` accessors."""
        return {role: self._as_ref(ref) for role, ref in self._raw.items() if ref is not None}

    @property
    def src_lin(self):
        """Source linear reference (resolved accessor), or ``None``."""
        return self._as_ref(self._raw["src_lin"])

    @property
    def src_quad(self):
        """Source quadratic reference (resolved accessor), or ``None``."""
        return self._as_ref(self._raw["src_quad"])

    @property
    def tgt_lin(self):
        """Target linear reference (resolved accessor), or ``None``."""
        return self._as_ref(self._raw["tgt_lin"])

    @property
    def tgt_quad(self):
        """Target quadratic reference (resolved accessor), or ``None``."""
        return self._as_ref(self._raw["tgt_quad"])

    @property
    def is_quadratic(self) -> bool:
        """Inferred regime: ``True`` when quadratic (GW) references are present."""
        return self._raw["src_quad"] is not None or self._raw["tgt_quad"] is not None
