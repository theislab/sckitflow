import numpy as np
from anndata import AnnData

from sc_flow.data._composite import MatchedData, NestedData
from sc_flow.data._manager import DataManager
from sc_flow.data.containers._categorical import CategoricalData
from sc_flow.data.containers._coupling import CouplingData
from sc_flow.data.containers._distribution import DistributionData
from sc_flow.data.containers._mixed_type import MixedTypeData
from sc_flow.data.containers._state import StateData


class TestDataManager:
    def test_datamanager_basic(self, adata: AnnData):
        """Test basic DataManager functionality."""
        manager = DataManager()
        distr_data: DistributionData = manager.get_distribution_data(adata)

        # DistributionData type and lengths
        n_obs = adata.n_obs
        assert isinstance(distr_data, DistributionData)
        assert len(distr_data) == n_obs
        assert isinstance(distr_data.state_data, StateData)
        assert isinstance(distr_data.groups_data, CategoricalData)
        assert len(distr_data.state_data) == n_obs
        assert len(distr_data.groups_data) == n_obs
        if distr_data.condition_data is not None:
            assert isinstance(distr_data.condition_data, MixedTypeData)
            assert len(distr_data.condition_data) == n_obs
        if distr_data.target_data is not None:
            assert isinstance(distr_data.target_data, MixedTypeData)
            assert len(distr_data.target_data) == n_obs

        # Slicing works
        idxs = np.random.choice(len(distr_data), min(5, len(distr_data)), replace=False)
        sliced = distr_data[idxs]
        assert len(sliced) == len(idxs)
        assert all(
            len(getattr(sliced, attr)) == len(idxs)
            for attr in ["state_data", "groups_data", "condition_data", "target_data"]
            if getattr(sliced, attr) is not None
        )

        # __repr__ contains key info
        repr_str = repr(distr_data)
        assert "DistributionData" in repr_str
        assert f"n_obs={len(distr_data)}" in repr_str

    def test_nested_data_leaves(self, adata: AnnData):
        """Check that NestedData leaves are MatchedData with proper lengths."""
        manager = DataManager()
        nested: NestedData = manager.compile_adata(adata)
        assert isinstance(nested, NestedData)

        def check_leaf(node):
            if isinstance(node, NestedData):
                for v in node.mapping.values():
                    check_leaf(v)
            else:
                assert isinstance(node, MatchedData)
                assert len(node.target) > 0
                if node.source is not None:
                    assert len(node.source) > 0

        check_leaf(nested)

    def test_source_key_with_control_values(self, adata: AnnData):
        """Verify that control values are correctly propagated."""
        manager = DataManager(
            conditions={
                "drug": ("drugA", "drugB"),
                "ko": ("koA", "koB"),
            },
            conditions_reps={
                "drug": "drug",
                "ko": "ko",
            },
            control_values_dict={"drug": "control", "ko": "control"},
        )
        key = manager.source_key
        assert isinstance(key, tuple)
        assert len(key) > 0

    def test_zero_length_slice(self, adata: AnnData):
        """Check that slicing with empty indices returns empty DistributionData."""
        manager = DataManager()
        distr_data: DistributionData = manager.get_distribution_data(adata)
        sliced = distr_data[np.array([], dtype=int)]
        assert len(sliced) == 0
        for attr in ["state_data", "groups_data", "condition_data", "target_data"]:
            val = getattr(sliced, attr)
            if val is not None:
                assert len(val) == 0

    def test_coupling_data_shapes(self, adata: AnnData):
        """Check that source and target coupling data are valid."""
        manager = DataManager()
        distr_data: DistributionData = manager.get_distribution_data(adata)
        source_coupling, target_coupling = distr_data.source_coupling_data, distr_data.target_coupling_data
        assert isinstance(source_coupling, CouplingData)
        assert isinstance(target_coupling, CouplingData)
        assert len(source_coupling) == len(distr_data)
        assert len(target_coupling) == len(distr_data)
