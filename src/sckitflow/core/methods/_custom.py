from collections.abc import Callable
from typing import Literal, TypeVar

T = TypeVar("T", bound=type)

__all__ = ["register_method"]


def register_method(
    name: str,
    *,
    category: Literal["flow", "general"] = "flow",
) -> Callable[[T], T]:
    # Imported here rather than at module scope: `sckitflow.core.methods` imports this
    # module, so a top-level import would be circular.
    from sckitflow.core._types import PredictionData
    from sckitflow.core.methods import METHODS_REGISTRY
    from sckitflow.core.methods._base import BaseMethod, GenerativeFlow

    if category == "flow":
        BaseClass = GenerativeFlow
        required_user_methods = ["step_fn", "predict"]
    elif category == "general":
        BaseClass = BaseMethod
        required_user_methods = ["train_step", "predict"]
    else:
        raise ValueError(f"Unsupported category: {category}")

    def decorator(user_cls: T) -> T:
        # Validate user class
        if not hasattr(user_cls, "module_cls"):
            raise TypeError(f"{user_cls.__name__} must define a 'module_cls' class attribute.")
        for meth in required_user_methods:
            if not hasattr(user_cls, meth):
                raise TypeError(f"{user_cls.__name__} must define a '{meth}' method.")

        # Build the class dictionary for the dynamically created class
        class_dict = {
            "_module_cls": user_cls.module_cls,
        }

        if category == "flow":
            class_dict["_default_solver_cls"] = getattr(user_cls, "default_solver_cls", None)

            # Delegate abstract methods to user implementations
            def compute_loss(self, node, *args, **kwargs):
                """Delegate to user's step_fn method."""
                return user_cls.step_fn(self, node, *args, **kwargs)

            def infer(self, node, *args, **kwargs):
                """Delegate to user's predict method."""
                return user_cls.predict(self, node, *args, **kwargs)

            class_dict["compute_loss"] = compute_loss
            class_dict["infer"] = infer

            # Override __init__ to call base __init__ first, then optionally call user's __init__
            def __init__(self, *args, **kwargs):
                prob_path_cls = getattr(user_cls, "probability_path_cls", None)
                if prob_path_cls is not None:
                    kwargs.setdefault("probability_path", prob_path_cls())
                # Call BaseClass.__init__ (the MRO will resolve correctly)
                super(RegisteredMethod, self).__init__(*args, **kwargs)
                # If user defined __init__, call it (with self)
                if hasattr(user_cls, "__init__") and user_cls.__init__ is not object.__init__:
                    user_cls.__init__(self, *args, **kwargs)

            class_dict["__init__"] = __init__

        elif category == "general":

            def __init__(self, *args, **kwargs):
                super(RegisteredMethod, self).__init__(*args, **kwargs)
                if hasattr(user_cls, "__init__") and user_cls.__init__ is not object.__init__:
                    user_cls.__init__(self, *args, **kwargs)

            class_dict["__init__"] = __init__

            # `BaseMethod` declares `compute_loss` and `infer` abstract, and a
            # "general" method overrides their public callers (`train_step`, `predict`)
            # outright -- but the abstract slots still have to be filled or the class
            # cannot be instantiated at all.
            def compute_loss(self, step_data, *args, **kwargs):
                raise NotImplementedError(
                    f"{user_cls.__name__} is registered as a 'general' method and implements "
                    "`train_step` directly, so `compute_loss` is never used."
                )

            def infer(self, step_data, *args, **kwargs):
                raise NotImplementedError(
                    f"{user_cls.__name__} is registered as a 'general' method and implements "
                    "`predict` directly, so `infer` is never used."
                )

            class_dict["compute_loss"] = compute_loss
            class_dict["infer"] = infer

            # Wrap predict output into PredictionData
            def predict(self, matched_distr, *args, **kwargs):
                raw_output = user_cls.predict(self, matched_distr, *args, **kwargs)
                if isinstance(raw_output, PredictionData):
                    return raw_output
                return PredictionData(X=raw_output, traj=None)

            class_dict["predict"] = predict

        # Create the new class inheriting from (user_cls, BaseClass)
        RegisteredMethod = type(
            user_cls.__name__,
            (user_cls, BaseClass),
            class_dict,
        )
        RegisteredMethod.__module__ = user_cls.__module__
        RegisteredMethod.__doc__ = user_cls.__doc__

        # Register in the method registry
        if name in METHODS_REGISTRY:
            raise ValueError(f"Method '{name}' already registered.")
        METHODS_REGISTRY[name] = RegisteredMethod

        return user_cls

    return decorator
