from collections.abc import Callable
from typing import Literal, TypeVar

T = TypeVar("T", bound=type)

__all__ = ["register_method"]


def register_method(
    name: str,
    backend: Literal["torch", "jax"] = "torch",
    *,
    category: Literal["flow", "general"] = "flow",
) -> Callable[[T], T]:
    # Backend‑specific imports
    if backend == "torch":
        from sc_flow.backends.torch._types import PredictionData as TorchPredictionData
        from sc_flow.backends.torch.methods import METHODS_REGISTRY
        from sc_flow.backends.torch.methods._base import TorchBaseMethod, TorchGenerativeFlow

        PredictionDataClass = TorchPredictionData

        if category == "flow":
            BaseClass = TorchGenerativeFlow
            required_user_methods = ["compute_loss", "predict"]
        elif category == "general":
            BaseClass = TorchBaseMethod
            required_user_methods = ["train_step", "predict"]
        else:
            raise ValueError(f"Unsupported category: {category}")
    elif backend == "jax":
        raise NotImplementedError("JAX backend not yet implemented")
    else:
        raise ValueError(f"Unsupported backend: {backend}")

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

            # Wrap predict output into PredictionData
            def predict(self, matched_distr, *args, **kwargs):
                raw_output = user_cls.predict(self, matched_distr, *args, **kwargs)
                if isinstance(raw_output, PredictionDataClass):
                    return raw_output
                if hasattr(raw_output, "samples"):
                    return raw_output
                return PredictionDataClass(samples=raw_output, traj=None)

            class_dict["predict"] = predict

        # Create the new class inheriting from (user_cls, BaseClass)
        RegisteredMethod = type(
            user_cls.__name__,
            (user_cls, BaseClass),
            class_dict,
        )
        RegisteredMethod.__module__ = user_cls.__module__
        RegisteredMethod.__doc__ = user_cls.__doc__

        # Register in the backend's method registry
        if name in METHODS_REGISTRY:
            raise ValueError(f"Method '{name}' already registered for backend '{backend}'.")
        METHODS_REGISTRY[name] = RegisteredMethod

        return user_cls

    return decorator
