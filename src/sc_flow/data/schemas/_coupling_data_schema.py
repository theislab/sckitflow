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
        if ref is None:
            return None
        if isinstance(ref, str):
            return A.X if ref == "X" else A.obsm[ref]
        return ref

    def _verify_args(self) -> None:
        r = self._raw
        if (r["src_lin"] is None) != (r["tgt_lin"] is None):
            raise ValueError("linear coupling needs both src_lin and tgt_lin (or neither).")
        if (r["src_quad"] is None) != (r["tgt_quad"] is None):
            raise ValueError("quadratic coupling needs both src_quad and tgt_quad (or neither).")
        if all(v is None for v in r.values()):
            raise ValueError("CouplingDataSchema needs at least one regime (lin and/or quad).")

    def _verify_schema(self, adata: AnnData) -> None:
        for role, ref in self.refs.items():
            if ref not in adata:
                raise KeyError(f"coupling {role} reference not found in adata.")

    @property
    def refs(self) -> Mapping[str, object]:
        return {role: self._as_ref(ref) for role, ref in self._raw.items() if ref is not None}

    @property
    def src_lin(self):
        return self._as_ref(self._raw["src_lin"])

    @property
    def src_quad(self):
        return self._as_ref(self._raw["src_quad"])

    @property
    def tgt_lin(self):
        return self._as_ref(self._raw["tgt_lin"])

    @property
    def tgt_quad(self):
        return self._as_ref(self._raw["tgt_quad"])

    @property
    def is_quadratic(self) -> bool:
        return self._raw["src_quad"] is not None or self._raw["tgt_quad"] is not None
