# sc_flow/backends/__init__.py
from collections.abc import Callable
from typing import Literal, TypeVar

T = TypeVar("T", bound=type)

__all__ = ["register_method"]


def register_method(
    name: str,
    backend: Literal["torch", "jax"] = "torch",
    *,
    category: Literal["flow", "general"] = "flow",
    allow_overwrite: bool = True,
) -> Callable[[T], T]:
    """Decorator to register a new method for a specific backend and category."""
    # ----------------------------------------------------------------------
    # 1. Determine backend‑specific imports and base classes
    # ----------------------------------------------------------------------
    if backend == "torch":
        from sc_flow.backends.torch._types import PredictionData as TorchPredictionData
        from sc_flow.backends.torch.methods import METHODS_REGISTRY
        from sc_flow.backends.torch.methods._base import TorchBaseMethod, TorchGenerativeFlow

        PredictionDataClass = TorchPredictionData

        if category == "flow":
            BaseClass = TorchGenerativeFlow
            required_user_methods = ["compute_loss", "predict"]
            init_hook_name = "__init_flow__"
        elif category == "general":
            BaseClass = TorchBaseMethod
            required_user_methods = ["train_step", "predict"]
            init_hook_name = "__init_method__"
        else:
            raise ValueError(f"Unsupported category: {category}")

    elif backend == "jax":
        # Placeholder for JAX
        raise NotImplementedError("JAX backend not yet implemented")
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    def decorator(user_cls: T) -> T:
        # ------------------------------------------------------------------
        # 2. Validate required attributes
        # ------------------------------------------------------------------
        if not hasattr(user_cls, "module_cls"):
            raise TypeError(f"{user_cls.__name__} must define a 'module_cls' class attribute.")

        for method_name in required_user_methods:
            if not hasattr(user_cls, method_name):
                raise TypeError(f"{user_cls.__name__} must define a '{method_name}' method.")

        # ------------------------------------------------------------------
        # 3. Dynamically create the wrapper class
        # ------------------------------------------------------------------
        class WrappedMethod(BaseClass):
            _module_cls = user_cls.module_cls

            if category == "flow":
                _default_solver_cls = getattr(user_cls, "default_solver_cls", None)

                def __init__(self, *args, **kwargs):
                    prob_path_cls = getattr(user_cls, "probability_path_cls", None)
                    if prob_path_cls is not None:
                        kwargs.setdefault("probability_path", prob_path_cls())
                    super().__init__(*args, **kwargs)
                    if hasattr(user_cls, init_hook_name):
                        getattr(user_cls, init_hook_name)(self, *args, **kwargs)

                def _compute_loss(self, step_data, *args, **kwargs):
                    return user_cls.compute_loss(self, step_data, *args, **kwargs)

                def _predict(self, step_data, *args, **kwargs):
                    # For flow methods, the user already returns PredictionData.
                    return user_cls.predict(self, step_data, *args, **kwargs)

            elif category == "general":

                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    if hasattr(user_cls, init_hook_name):
                        getattr(user_cls, init_hook_name)(self, *args, **kwargs)

                def train_step(self, matched_distr, *args, **kwargs):
                    return user_cls.train_step(self, matched_distr, *args, **kwargs)

                def predict(self, matched_distr, *args, **kwargs):
                    raw_output = user_cls.predict(self, matched_distr, *args, **kwargs)
                    # Automatically wrap into PredictionData if needed
                    if isinstance(raw_output, PredictionDataClass):
                        return raw_output
                    # Check if it already has a 'samples' attribute (e.g., a duck-typed PredictionData)
                    if hasattr(raw_output, "samples"):
                        return raw_output
                    # Otherwise, assume it's the samples tensor/array and wrap it
                    return PredictionDataClass(samples=raw_output, traj=None)

        # Preserve original metadata
        WrappedMethod.__name__ = user_cls.__name__
        WrappedMethod.__doc__ = user_cls.__doc__
        WrappedMethod.__module__ = user_cls.__module__

        # ------------------------------------------------------------------
        # 4. Register in the backend's method registry
        # ------------------------------------------------------------------
        if name in METHODS_REGISTRY and not allow_overwrite:
            raise ValueError(f"Method '{name}' is already registered for backend '{backend}'.")
        METHODS_REGISTRY[name] = WrappedMethod

        return user_cls

    return decorator
