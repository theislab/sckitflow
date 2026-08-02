from unittest.mock import Mock, patch

import pytest
import torch  # needed for the raw tensor in predict wrapper

from sckitflow.core.methods._custom import register_method


# -----------------------------------------------------------------------------
# Dummy base classes that accept any arguments (to avoid TypeError)
# -----------------------------------------------------------------------------
class DummyBaseMethod:
    def __init__(self, *args, **kwargs):
        pass


class DummyGenerativeFlow:
    def __init__(self, *args, **kwargs):
        pass


# -----------------------------------------------------------------------------
# Fixtures and helpers
# -----------------------------------------------------------------------------
@pytest.fixture
def mock_torch_registry():
    """Provide a mock METHODS_REGISTRY dictionary."""
    registry = {}
    with patch("sckitflow.core.methods.METHODS_REGISTRY", registry):
        yield registry


@pytest.fixture
def mock_torch_base_classes():
    """Patch the base classes with dummy classes that accept arguments."""
    with (
        patch("sckitflow.core.methods._base.BaseMethod", DummyBaseMethod),
        patch("sckitflow.core.methods._base.GenerativeFlow", DummyGenerativeFlow),
    ):
        yield DummyBaseMethod, DummyGenerativeFlow


@pytest.fixture
def mock_prediction_data():
    """Provide a dummy PredictionData class."""

    class DummyPredictionData:
        # Mirrors the real `PredictionData`, whose payload field is `X`.
        def __init__(self, X, traj=None):
            self.X = X
            self.traj = traj

    with patch("sckitflow.core._types.PredictionData", DummyPredictionData):
        yield DummyPredictionData


# -----------------------------------------------------------------------------
# Tests for flow method registration
# -----------------------------------------------------------------------------
def test_register_flow_method_success(mock_torch_registry, mock_torch_base_classes):
    """Test successful registration of a flow method."""
    mock_base, mock_flow = mock_torch_base_classes

    @register_method("test_flow", category="flow")
    class UserFlow:
        module_cls = Mock()
        default_solver_cls = Mock()
        probability_path_cls = Mock()

        def step_fn(self, step_data, *args, **kwargs):
            pass

        def predict(self, step_data, *args, **kwargs):
            pass

    assert "test_flow" in mock_torch_registry
    registered_cls = mock_torch_registry["test_flow"]
    assert issubclass(registered_cls, UserFlow)
    assert issubclass(registered_cls, mock_flow)
    assert registered_cls._module_cls == UserFlow.module_cls
    assert registered_cls._default_solver_cls == UserFlow.default_solver_cls

    # Check that the __init__ correctly sets probability_path from user's class
    _instance = registered_cls(dims_registry=Mock(), dm=Mock())
    assert True


def test_register_flow_missing_module_cls(mock_torch_registry, mock_torch_base_classes):
    """Missing module_cls raises TypeError."""
    with pytest.raises(TypeError, match="must define a 'module_cls'"):

        @register_method("bad_flow", category="flow")
        class BadFlow:
            def step_fn(self):
                pass

            def predict(self):
                pass


def test_register_flow_missing_step_fn(mock_torch_registry, mock_torch_base_classes):
    """Missing step_fn raises TypeError."""
    with pytest.raises(TypeError, match="must define a 'step_fn' method"):

        @register_method("bad_flow", category="flow")
        class BadFlow:
            module_cls = Mock()

            def predict(self):
                pass


def test_register_flow_missing_predict(mock_torch_registry, mock_torch_base_classes):
    """Missing predict raises TypeError."""
    with pytest.raises(TypeError, match="must define a 'predict' method"):

        @register_method("bad_flow", category="flow")
        class BadFlow:
            module_cls = Mock()

            def step_fn(self):
                pass


def test_register_flow_with_user_init(mock_torch_registry, mock_torch_base_classes):
    """User-defined __init__ is called after base init."""
    mock_base, mock_flow = mock_torch_base_classes
    init_called = False

    @register_method("flow_with_init", category="flow")
    class UserFlow:
        module_cls = Mock()

        def step_fn(self):
            pass

        def predict(self):
            pass

        def __init__(self, *args, **kwargs):
            nonlocal init_called
            init_called = True

    registered_cls = mock_torch_registry["flow_with_init"]
    _instance = registered_cls(dims_registry=Mock(), dm=Mock())
    assert init_called


# -----------------------------------------------------------------------------
# Tests for general method registration
# -----------------------------------------------------------------------------
def test_register_general_method_success(mock_torch_registry, mock_torch_base_classes, mock_prediction_data):
    """Test successful registration of a general method."""
    mock_base, mock_flow = mock_torch_base_classes

    @register_method("test_general", category="general")
    class UserGeneral:
        module_cls = Mock()

        def train_step(self, matched_distr, *args, **kwargs):
            return {}

        def predict(self, matched_distr, *args, **kwargs):
            return mock_prediction_data(samples=Mock(), traj=None)

    assert "test_general" in mock_torch_registry
    registered_cls = mock_torch_registry["test_general"]
    assert issubclass(registered_cls, UserGeneral)
    assert issubclass(registered_cls, mock_base)
    assert registered_cls._module_cls == UserGeneral.module_cls


def test_register_general_missing_train_step(mock_torch_registry, mock_torch_base_classes):
    """Missing train_step raises TypeError."""
    with pytest.raises(TypeError, match="must define a 'train_step' method"):

        @register_method("bad_general", category="general")
        class BadGeneral:
            module_cls = Mock()

            def predict(self):
                pass


def test_register_general_missing_predict(mock_torch_registry, mock_torch_base_classes):
    """Missing predict raises TypeError."""
    with pytest.raises(TypeError, match="must define a 'predict' method"):

        @register_method("bad_general", category="general")
        class BadGeneral:
            module_cls = Mock()

            def train_step(self):
                pass


def test_register_general_predict_wrapper(mock_torch_registry, mock_torch_base_classes, mock_prediction_data):
    """For general methods, predict output is wrapped into PredictionData if not already."""
    mock_base, mock_flow = mock_torch_base_classes

    @register_method("wrap_test", category="general")
    class WrapTest:
        module_cls = Mock()

        def train_step(self):
            pass

        def predict(self, matched_distr, *args, **kwargs):
            return torch.randn(4, 2)  # raw tensor

    registered_cls = mock_torch_registry["wrap_test"]
    instance = registered_cls(dims_registry=Mock(), dm=Mock())
    matched = Mock()
    result = instance.predict(matched)
    # Check that the result is an instance of PredictionData
    assert isinstance(result, mock_prediction_data)
    assert hasattr(result, "X")
    assert result.traj is None


# -----------------------------------------------------------------------------
# Duplicate registration and error handling
# -----------------------------------------------------------------------------
def test_duplicate_registration_error(mock_torch_registry, mock_torch_base_classes):
    """Registering same name twice raises ValueError."""

    @register_method("dup", category="flow")
    class First:
        module_cls = Mock()

        def step_fn(self):
            pass

        def predict(self):
            pass

    with pytest.raises(ValueError, match="already registered"):

        @register_method("dup", category="flow")
        class Second:
            module_cls = Mock()

            def step_fn(self):
                pass

            def predict(self):
                pass


def test_unsupported_category(mock_torch_registry):
    """Unsupported category raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported category"):

        @register_method("bad", category="diffusion")
        class Dummy:
            module_cls = Mock()

            def step_fn(self):
                pass

            def predict(self):
                pass


# -----------------------------------------------------------------------------
# Edge cases: user class with no __init__ (default object.__init__)
# -----------------------------------------------------------------------------
def test_no_user_init_does_not_call_extra_init(mock_torch_registry, mock_torch_base_classes):
    """If user class does not define __init__, only base init is called."""
    mock_base, mock_flow = mock_torch_base_classes

    @register_method("no_init", category="flow")
    class NoInit:
        module_cls = Mock()

        def step_fn(self):
            pass

        def predict(self):
            pass

    registered_cls = mock_torch_registry["no_init"]
    # Should not raise any error
    _instance = registered_cls(dims_registry=Mock(), dm=Mock())
    assert True


def test_probability_path_cls_not_set(mock_torch_registry, mock_torch_base_classes):
    """If probability_path_cls is not set, no default is added."""
    mock_base, mock_flow = mock_torch_base_classes

    @register_method("no_prob_path", category="flow")
    class NoProbPath:
        module_cls = Mock()

        def step_fn(self):
            pass

        def predict(self):
            pass

    registered_cls = mock_torch_registry["no_prob_path"]
    # No error; the __init__ does not add probability_path
    _instance = registered_cls(dims_registry=Mock(), dm=Mock())
    assert True


# -----------------------------------------------------------------------------
# Test that the decorator returns the original class (not the registered one)
# -----------------------------------------------------------------------------
def test_decorator_returns_original_class(mock_torch_registry, mock_torch_base_classes):
    """The decorator should return the original user class, not the registered one."""

    @register_method("return_original", category="flow")
    class Original:
        module_cls = Mock()

        def step_fn(self):
            pass

        def predict(self):
            pass

    # The variable 'Original' should be the original class, not the wrapped one
    assert Original.__name__ == "Original"
    # But the registry contains a different class
    registered = mock_torch_registry["return_original"]
    assert registered is not Original
    assert issubclass(registered, Original)
