from collections.abc import Callable, Collection
from pathlib import Path
from typing import Any

__all__ = ["ExternalModelContext"]


class ExternalModelContext:
    """Lazy loader for external models with flexible loading and calling.

    The model is loaded only when first accessed (by :meth:`load` or when calling the instance).
    Custom loaders, post‑load processing, and forward method selection are supported.

    :param model_path: Path to the serialized model (file or directory).
    :param model_load_args: Positional arguments for the loader.
    :param model_load_kwargs: Keyword arguments for the loader.
    :param forward_method: Name of the method to call on the loaded object
        (if None, the object itself is called).
    :param forward_args: Positional arguments to prepend to the forward call.
    :param forward_kwargs: Keyword arguments for the forward call.
    :param post_load_callable: Function to prepare the loaded model
        (signature: (model, **post_load_kwargs) -> prepared_model).
    :param post_load_kwargs: Keyword arguments passed to the post_load_callable.
    :param loader: Optional custom loader (if None, inferred from extension).
    :param allow_missing_forward_method: If True, do not check existence of forward_method.
    """

    _EXTENSION_LOADERS: dict[str, Callable[[Path, Collection[Any], dict[str, Any]], Any]] = {}

    @classmethod
    def register_loader(cls, extension: str, loader: Callable) -> None:
        """Register a loader for a given file extension (e.g., '.onnx')."""
        cls._EXTENSION_LOADERS[extension.lower()] = loader

    def __init__(
        self,
        model: Any | None = None,
        model_path: str | None = None,
        model_load_args: Collection[Any] = (),
        model_load_kwargs: dict[str, Any] = None,
        forward_method: str | None = None,
        forward_args: Collection[Any] = (),
        forward_kwargs: dict[str, Any] = None,
        post_load_callable: Callable[..., Any] | None = None,
        post_load_kwargs: dict[str, Any] = None,
        loader: Callable[[Path, Collection[Any], dict[str, Any]], Any] | None = None,
        allow_missing_forward_method: bool = True,
        runtime_args_first: bool = True,
    ) -> None:
        if model is None and model_path is None:
            raise ValueError("At least one of model or model_path should be passed")

        self._model: Any = model
        self._model_path = model_path
        self._model_load_args = model_load_args
        self._model_load_kwargs = model_load_kwargs or {}
        self._forward_method = forward_method
        self._forward_args = forward_args
        self._forward_kwargs = forward_kwargs or {}
        self._post_load_callable = post_load_callable
        self._post_load_kwargs = post_load_kwargs or {}
        self._loader = loader
        self._allow_missing_forward_method = allow_missing_forward_method
        self._runtime_args_first = runtime_args_first

    def _default_loader(self, path: Path, args: Collection[Any], kwargs: dict[str, Any]) -> Any:
        """Infer loader from file extension and load the model."""
        ext = path.suffix.lower()
        # Check registered loaders first
        if ext in self._EXTENSION_LOADERS:
            return self._EXTENSION_LOADERS[ext](path, args, kwargs)

        # Built‑in loaders
        if ext in (".pth", ".pt"):
            import torch

            return torch.load(path, *args, **kwargs)
        elif ext == ".pkl":
            import pickle

            with open(path, "rb") as f:
                return pickle.load(f, *args, **kwargs)
        elif ext == ".joblib":
            import joblib

            return joblib.load(path, *args, **kwargs)
        elif ext in (".h5", ".hdf5"):
            import h5py

            return h5py.File(path, *args, **kwargs)
        else:
            import cloudpickle

            with open(path, "rb") as f:
                return cloudpickle.load(f, *args, **kwargs)

    def _get_loader(self) -> Callable[[Path, Collection[Any], dict[str, Any]], Any]:
        return self._loader if self._loader is not None else self._default_loader

    def load(self) -> Any:
        """Load the model and apply post-load preparation (if any)."""
        if self._model is None:
            path = Path(self._model_path)
            if not path.exists():
                raise FileNotFoundError(f"Model path does not exist: {path}")

            loader = self._get_loader()
            raw_model = loader(path, self._model_load_args, self._model_load_kwargs)
            # Apply post‑load preparation
            if self._post_load_callable is not None:
                self._model = self._post_load_callable(raw_model, **self._post_load_kwargs)
            else:
                self._model = raw_model
        return self._model

    def is_loaded(self) -> bool:
        return self._model is not None

    def reload(self) -> Any:
        self._model = None
        return self.load()

    def unload(self) -> None:
        self._model = None

    def _get_forward_callable(self) -> Any:
        obj = self.load()
        if self._forward_method is None:
            if not callable(obj):
                raise TypeError(f"Loaded object is not callable and no forward_method specified: {type(obj)}")
            return obj
        else:
            if not hasattr(obj, self._forward_method):
                if self._allow_missing_forward_method:
                    return obj
                raise AttributeError(f"Loaded object has no attribute '{self._forward_method}'")
            return getattr(obj, self._forward_method)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the model with the forward_args/kwargs merged with call‑time arguments."""
        forward_fn = self._get_forward_callable()
        if self._runtime_args_first:
            final_args = list(args) + list(self._forward_args)
        else:
            final_args = list(self._forward_args) + list(args)
        final_kwargs = {**self._forward_kwargs, **kwargs}
        return forward_fn(*final_args, **final_kwargs)

    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded() else "not loaded"
        return f"{self.__class__.__name__}(path={self._model_path!r}, status={status})"

    @property
    def model(self) -> Any | None:
        """Returns the underlying model."""
        return self._model

    @model.setter
    def model(self, model: Any) -> None:
        """Sets the underlying model."""
        self._model = model
