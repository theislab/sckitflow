import pytest
from anndata.acc import A

from sckitflow.data.schemas import CouplingDataSchema

LIN, SRC, TGT = A.obsm["X_repr"], A.obsm["X_src"], A.obsm["X_tgt"]


@pytest.mark.parametrize("rep_lin", [None, LIN, A.X, A.layers["lin"]])
@pytest.mark.parametrize("source_rep_quad", [None, SRC, A.layers["src"]])
@pytest.mark.parametrize("target_rep_quad", [None, TGT])
def test_terms(rep_lin, source_rep_quad, target_rep_quad) -> None:
    kwargs = {"rep_lin": rep_lin, "source_rep_quad": source_rep_quad, "target_rep_quad": target_rep_quad}
    if rep_lin is None and (source_rep_quad is None) != (target_rep_quad is None):
        with pytest.raises(ValueError, match="both `source_rep_quad` and `target_rep_quad`"):
            CouplingDataSchema(**kwargs)
        return
    schema = CouplingDataSchema(**kwargs)
    source = {"source_coupling_lin": rep_lin, "source_coupling_quad": source_rep_quad}
    target = {"target_coupling_lin": rep_lin, "target_coupling_quad": target_rep_quad}
    assert schema.source_terms == {t: r for t, r in source.items() if r is not None}
    assert schema.target_terms == {t: r for t, r in target.items() if r is not None}


def test_one_rep_serves_the_linear_and_a_quadratic_term() -> None:
    schema = CouplingDataSchema(rep_lin=LIN, source_rep_quad=LIN, target_rep_quad=TGT)
    assert schema.source_terms == {"source_coupling_lin": LIN, "source_coupling_quad": LIN}
