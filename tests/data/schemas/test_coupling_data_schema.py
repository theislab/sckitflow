import pytest
from anndata.acc import A

from sckitflow.data.schemas import CouplingDataSchema

LIN, SRC, TGT = A.obsm["X_repr"], A.obsm["X_src"], A.obsm["X_tgt"]


class TestCouplingDataSchema:
    @pytest.mark.parametrize("rep_lin", [None, LIN, A.X, A.layers["lin"]])
    @pytest.mark.parametrize("source_rep_quad", [None, SRC, A.layers["src"]])
    @pytest.mark.parametrize("target_rep_quad", [None, TGT])
    def test_init(self, rep_lin, source_rep_quad, target_rep_quad) -> None:
        kwargs = {"rep_lin": rep_lin, "source_rep_quad": source_rep_quad, "target_rep_quad": target_rep_quad}
        if rep_lin is None and (source_rep_quad is not None or target_rep_quad is not None):
            with pytest.raises(ValueError, match="needs `rep_lin`"):
                CouplingDataSchema(**kwargs)
            return
        schema = CouplingDataSchema(**kwargs)
        assert schema.rep_lin == rep_lin
        assert schema.source_rep_quad == source_rep_quad
        assert schema.target_rep_quad == target_rep_quad
        assert schema.use_linear is True


class TestUseLinearFalse:
    """Pure Gromov-Wasserstein: no shared linear term, both quadratics required."""

    def test_init_rejects_rep_lin(self) -> None:
        with pytest.raises(ValueError, match="redundant"):
            CouplingDataSchema(rep_lin=LIN, source_rep_quad=SRC, target_rep_quad=TGT, use_linear=False)

    @pytest.mark.parametrize("source_rep_quad, target_rep_quad", [(None, TGT), (SRC, None), (None, None)])
    def test_init_requires_both_quads(self, source_rep_quad, target_rep_quad) -> None:
        with pytest.raises(ValueError, match="requires both"):
            CouplingDataSchema(source_rep_quad=source_rep_quad, target_rep_quad=target_rep_quad, use_linear=False)

    def test_init_valid(self) -> None:
        schema = CouplingDataSchema(source_rep_quad=SRC, target_rep_quad=TGT, use_linear=False)
        assert schema.use_linear is False
        assert schema.rep_lin is None
