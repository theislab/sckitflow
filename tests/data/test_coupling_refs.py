"""Unit tests for :class:`~sc_flow.core.data.schemas.CouplingDataSchema` (schema-generalization Change 3).

Regime is inferred from which ``*_lin`` / ``*_quad`` refs are present; refs may be ``anndata.acc``
accessors or bare strings, and both resolve/validate the same way.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
from anndata.acc import A

from sc_flow.core.data.schemas import CouplingDataSchema


def _adata() -> ad.AnnData:
    a = ad.AnnData(X=np.zeros((6, 3), dtype="float32"))
    a.obsm["X_lin"] = np.zeros((6, 4), dtype="float32")
    a.obsm["X_quad"] = np.zeros((6, 5), dtype="float32")
    return a


def test_linear_regime_inferred():
    s = CouplingDataSchema(src_lin="X_lin", tgt_lin="X_lin")
    assert not s.is_quadratic
    assert set(s.refs) == {"src_lin", "tgt_lin"}


def test_quadratic_regime_inferred():
    s = CouplingDataSchema(src_lin="X_lin", tgt_lin="X_lin", src_quad="X_quad", tgt_quad="X_quad")
    assert s.is_quadratic
    assert set(s.refs) == {"src_lin", "tgt_lin", "src_quad", "tgt_quad"}


def test_asymmetric_linear_rejected():
    with pytest.raises(ValueError, match="linear"):
        CouplingDataSchema(src_lin="X_lin")  # missing tgt_lin


def test_asymmetric_quadratic_rejected():
    with pytest.raises(ValueError, match="quadratic"):
        CouplingDataSchema(src_lin="X_lin", tgt_lin="X_lin", src_quad="X_quad")  # missing tgt_quad


def test_empty_rejected():
    with pytest.raises(ValueError, match="at least one regime"):
        CouplingDataSchema()


def test_str_and_accessor_refs_both_validate():
    adata = _adata()
    CouplingDataSchema(src_lin="X_lin", tgt_lin="X_lin")._verify_schema(adata)
    CouplingDataSchema(src_lin=A.obsm["X_lin"], tgt_lin=A.obsm["X_lin"])._verify_schema(adata)


def test_missing_ref_rejected_by_verify_schema():
    with pytest.raises(KeyError, match="src_lin"):
        CouplingDataSchema(src_lin="nope", tgt_lin="X_lin")._verify_schema(_adata())
