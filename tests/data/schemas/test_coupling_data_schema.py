import numpy as np
import pytest
from anndata import AnnData

from sc_flow.data.containers import CouplingData, StateData
from sc_flow.data.schemas import CouplingDataSchema

invalid_key = "invalid_key"


class TestCouplingDataSchema:
    @staticmethod
    def _is_valid_args(
        source_rep: str | None,
        target_rep: str | None,
        n_shared_dims: int | None,
    ) -> bool:
        if n_shared_dims is not None:
            if n_shared_dims <= 0:
                return False
            if target_rep == source_rep:
                return False
        return True

    @staticmethod
    def _verify_coupling_data(
        data: CouplingData,
        data_array: np.ndarray,
        n_shared_dims: int | None,
    ) -> None:
        assert isinstance(data, CouplingData)
        tot_dims = data.state_lin.X.shape[1]
        if n_shared_dims is None:
            assert data.state_quad is None
        else:
            tot_dims = tot_dims + data.state_quad.X.shape[1]
            assert isinstance(data.state_quad, StateData)
        assert data_array.shape[1] == tot_dims

    @pytest.mark.parametrize("source_rep", [None, "X_src", invalid_key])
    @pytest.mark.parametrize("target_rep", [None, "X_tgt", invalid_key])
    @pytest.mark.parametrize("n_shared_dims", [None, 0, 10])
    def test_init(
        self,
        source_rep: str | None,
        target_rep: str | None,
        n_shared_dims: int | None,
    ) -> None:
        # verify arguments
        is_valid_args = self._is_valid_args(source_rep, target_rep, n_shared_dims)
        if not is_valid_args:
            with pytest.raises(ValueError):
                schema = CouplingDataSchema(source_rep=source_rep, target_rep=target_rep, n_shared_dims=n_shared_dims)
            return None

        # initialize schema
        schema = CouplingDataSchema(source_rep=source_rep, target_rep=target_rep, n_shared_dims=n_shared_dims)
        if n_shared_dims is None:
            assert not schema.has_incomparable_spaces
        else:
            assert schema.has_incomparable_spaces

    @pytest.mark.parametrize("source_rep", [None, "X_src", invalid_key])
    @pytest.mark.parametrize("target_rep", [None, "X_tgt", invalid_key])
    @pytest.mark.parametrize("n_shared_dims", [None, 10, 16])
    def test_get_data(
        self,
        adata: AnnData,
        source_rep: str | None,
        target_rep: str | None,
        n_shared_dims: int | None,
    ) -> None:
        # skip invalid arguments
        is_valid_args = self._is_valid_args(source_rep, target_rep, n_shared_dims)
        if not is_valid_args:
            pytest.skip()

        # initialize schema
        schema = CouplingDataSchema(source_rep=source_rep, target_rep=target_rep, n_shared_dims=n_shared_dims)

        # wrong identifier
        if source_rep == invalid_key and n_shared_dims is not None:
            with pytest.raises(KeyError):
                _data = schema.get_data(adata)
            return None
        if target_rep == invalid_key:
            with pytest.raises(KeyError):
                _data = schema.get_data(adata)
            return None

        # extract data
        target_data = schema.extract_array(adata, target_rep)
        if n_shared_dims is not None:
            source_data = schema.extract_array(adata, source_rep)
        else:
            source_data = target_data

        # wrong dimensions
        if n_shared_dims is not None:
            if (n_shared_dims >= source_data.shape[1]) or (n_shared_dims >= target_data.shape[1]):
                with pytest.raises(ValueError):
                    _data = schema.get_data(adata)
                return None

        source_coupling, target_coupling = schema.get_data(adata)

        self._verify_coupling_data(source_coupling, source_data, n_shared_dims)
        self._verify_coupling_data(target_coupling, target_data, n_shared_dims)
        source_coupling.assert_same_spatial_dims(target_coupling)
