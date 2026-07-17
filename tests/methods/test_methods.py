from unittest.mock import Mock

import pytest

from sc_flow.data._dims_registry import DataDimensionalitiesRegistry
from sc_flow.data._manager import DataManager
from sc_flow.methods._methods import BaseGenerativeFlow, BaseMethod


# -----------------------------------------------------------------------------
# Dummy concrete classes for testing abstract bases
# -----------------------------------------------------------------------------
class ConcreteMethod(BaseMethod):
    """Concrete implementation of BaseMethod for testing."""

    def build_module(self, *args, **kwargs):
        return Mock()

    def set_train_mode(self, mode: bool) -> None:
        self._train_mode = mode

    def train_step(self, *args, **kwargs):
        return 0, {"loss": 0.0}

    def predict(self, *args, **kwargs):
        return None


class ConcreteGenerativeFlow(BaseGenerativeFlow):
    """Concrete implementation of BaseGenerativeFlow for testing."""

    _default_solver_cls = Mock()

    def build_module(self, *args, **kwargs):
        return Mock()

    def set_train_mode(self, mode: bool) -> None:
        pass

    def train_step(self, *args, **kwargs):
        return 0, {}

    def predict(self, *args, **kwargs):
        return None


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def mock_dims_registry():
    registry = Mock(spec=DataDimensionalitiesRegistry)
    registry.feature_names = ["gene1", "gene2"]
    registry.n_features = 2
    return registry


@pytest.fixture
def mock_data_manager():
    dm = Mock(spec=DataManager)
    dm.control_values_dict = None
    return dm


# -----------------------------------------------------------------------------
# Test suite for BaseMethod
# -----------------------------------------------------------------------------
class TestBaseMethod:
    """Tests for the abstract BaseMethod class."""

    def test_abstract_class_cannot_be_instantiated(self):
        """BaseMethod should raise TypeError if abstract methods not implemented."""
        with pytest.raises(TypeError):
            BaseMethod(dims_registry=Mock(), dm=Mock(), is_paired_setting=False)

    def test_concrete_method_instantiation(self, mock_dims_registry, mock_data_manager):
        """Concrete subclass should be instantiable and set attributes correctly."""
        method = ConcreteMethod(mock_dims_registry, mock_data_manager, False)
        assert method._dims_registry is mock_dims_registry
        assert method._dm is mock_data_manager
        assert method._is_paired_setting is False
        assert method._module is not None  # built by build_module

    def test_properties_return_correct_values(self, mock_dims_registry, mock_data_manager):
        method = ConcreteMethod(mock_dims_registry, mock_data_manager, True)
        assert method.module is not None
        assert method.dm is mock_data_manager
        assert method.dims_registry is mock_dims_registry
        assert method.is_paired_setting is True

    def test_set_train_mode(self, mock_dims_registry, mock_data_manager):
        method = ConcreteMethod(mock_dims_registry, mock_data_manager, False)
        method.set_train_mode(True)
        assert method._train_mode is True
        method.set_train_mode(False)
        assert method._train_mode is False


# -----------------------------------------------------------------------------
# Test suite for BaseGenerativeFlow
# -----------------------------------------------------------------------------
class TestBaseGenerativeFlow:
    """Tests for the abstract BaseGenerativeFlow class."""

    def test_init_defaults(self, mock_dims_registry, mock_data_manager):
        """Default attributes should be None when not provided."""
        flow = ConcreteGenerativeFlow(mock_dims_registry, mock_data_manager, False)
        assert flow._probability_path is None
        assert flow._match_fn is None
        assert flow._noise_sampler is None
        assert flow._time_sampler is None
        assert flow.generate_from_noise is True

    def test_generate_from_noise_forced_when_unpaired(self, mock_dims_registry, mock_data_manager):
        """When is_paired_setting=False, generate_from_noise should be forced to True."""
        flow = ConcreteGenerativeFlow(
            mock_dims_registry,
            mock_data_manager,
            False,
            generate_from_noise=False,  # user tries to set False
        )
        # In BaseGenerativeFlow.__init__, if not paired, generate_from_noise becomes True
        assert flow.generate_from_noise is True

    def test_generate_from_noise_respected_when_paired(self, mock_dims_registry, mock_data_manager):
        """When paired, generate_from_noise can be set to False."""
        flow = ConcreteGenerativeFlow(mock_dims_registry, mock_data_manager, True, generate_from_noise=False)
        assert flow.generate_from_noise is False

    def test_properties_return_assigned_values(self, mock_dims_registry, mock_data_manager):
        prob_path = Mock()
        match_fn = Mock()
        noise_sampler = Mock()
        time_sampler = Mock()
        flow = ConcreteGenerativeFlow(
            mock_dims_registry,
            mock_data_manager,
            True,
            probability_path=prob_path,
            match_fn=match_fn,
            noise_sampler=noise_sampler,
            time_sampler=time_sampler,
            generate_from_noise=True,
        )
        assert flow.probability_path is prob_path
        assert flow.match_fn is match_fn
        assert flow.noise_sampler is noise_sampler
        assert flow.time_sampler is time_sampler
        assert flow.generate_from_noise is True
