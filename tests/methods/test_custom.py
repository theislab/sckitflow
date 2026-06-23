from unittest.mock import Mock

import pytest

from sc_flow.backends.torch.methods import METHODS_REGISTRY
from sc_flow.methods._custom import register_method
from sc_flow.methods._methods import BaseGenerativeFlow


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def dummy_step_fn(self, *args, **kwargs):
    return 0, {}


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear the backend registry before each test to ensure isolation."""
    METHODS_REGISTRY.clear()


# -----------------------------------------------------------------------------
# User‑defined flow classes
# -----------------------------------------------------------------------------
class UserFlow(BaseGenerativeFlow):
    module_cls = Mock()  # required by decorator (no underscore)
    _default_solver_cls = Mock()

    def predict(self, *args, **kwargs):
        return None

    step_fn = dummy_step_fn


class UserFlowMissingStepFn(BaseGenerativeFlow):
    """Missing step_fn – should cause a TypeError."""

    module_cls = Mock()
    _default_solver_cls = Mock()

    def predict(self, *args, **kwargs):
        return None


# For missing predict: a class that does NOT inherit predict from any parent
class UserFlowMissingPredict:
    """Not inheriting from BaseGenerativeFlow – predict is missing."""

    module_cls = Mock()

    step_fn = dummy_step_fn


class First(BaseGenerativeFlow):
    module_cls = Mock()
    _default_solver_cls = Mock()

    def predict(self, *args, **kwargs):
        pass

    step_fn = dummy_step_fn


class NoInit(BaseGenerativeFlow):
    module_cls = Mock()
    _default_solver_cls = Mock()

    def predict(self, *args, **kwargs):
        pass

    step_fn = dummy_step_fn


class Original(BaseGenerativeFlow):
    module_cls = Mock()
    _default_solver_cls = Mock()

    def predict(self, *args, **kwargs):
        pass

    step_fn = dummy_step_fn


class UserInitFlow(BaseGenerativeFlow):
    module_cls = Mock()
    _default_solver_cls = Mock()

    def __init__(self, dims_registry, dm, is_paired_setting, *args, extra=None, **kwargs):
        super().__init__(dims_registry, dm, is_paired_setting, *args, **kwargs)
        self.extra = extra

    def predict(self, *args, **kwargs):
        pass

    step_fn = dummy_step_fn


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
class TestCustomMethodRegistration:
    def test_register_method_success(self):
        """A class with all required members registers a subclass in the registry."""
        register_method("myflow")(UserFlow)
        assert "myflow" in METHODS_REGISTRY
        # The stored class is a dynamically created subclass, not the original
        assert issubclass(METHODS_REGISTRY["myflow"], UserFlow)

    def test_register_flow_missing_step_fn(self):
        """Missing step_fn should raise TypeError."""
        with pytest.raises(TypeError, match=r"must define a 'step_fn' method"):
            register_method("nostep")(UserFlowMissingStepFn)

    def test_register_flow_missing_predict(self):
        """If predict is not present in the MRO, raise TypeError."""
        with pytest.raises(TypeError, match=r"must define a 'predict' method"):
            register_method("nopredict")(UserFlowMissingPredict)

    def test_register_flow_with_user_init(self):
        """User class with custom __init__ still registers and extra args are passed."""
        register_method("userinit")(UserInitFlow)
        entry = METHODS_REGISTRY["userinit"]
        mock_dims = Mock()
        mock_dm = Mock()
        instance = entry(mock_dims, mock_dm, True, extra="value")
        assert instance.extra == "value"

    def test_duplicate_registration_error(self):
        """Registering the same name twice raises ValueError."""
        register_method("dup")(First)
        with pytest.raises(ValueError, match=r"already registered"):
            register_method("dup")(First)

    def test_no_user_init_does_not_call_extra_init(self):
        """
        When the user class doesn't override __init__, the registered
        class is still a subclass of it (the decorator's generated __init__
        does not call a user __init__ beyond the base).
        """
        register_method("noinit")(NoInit)
        assert issubclass(METHODS_REGISTRY["noinit"], NoInit)

    def test_decorator_returns_original_class(self):
        """The decorator must return the original class, not a new wrapper."""
        decorated = register_method("ret")(Original)
        assert decorated is Original
