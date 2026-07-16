"""Resolve ``{"kind": ..., **kwargs}`` specs into live objects.

YAML/dict configs describe objects (probability paths, solvers, transforms,
callbacks, metrics) by a string ``kind`` plus kwargs. These registries turn
those specs into instances/classes during :meth:`SCFlow.from_config`, keeping
the config fully serialisable. The fluent builder may bypass this and pass real
instances directly.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = [
    "resolve_probability_path",
    "resolve_flow_solver_cls",
    "resolve_metrics",
    "resolve_callbacks",
]


def _import_class(class_path: str) -> type:
    """Import a class from a ``"pkg.module.Class"`` or ``"pkg.module:Class"`` path."""
    import importlib

    module_path, sep, cls_name = class_path.rpartition(":" if ":" in class_path else ".")
    if not sep:
        raise ValueError(f"Invalid class_path {class_path!r}; expected 'pkg.module.Class'.")
    return getattr(importlib.import_module(module_path), cls_name)


def _class_path_spec(spec: Any) -> tuple[type, dict[str, Any]] | None:
    """Return ``(cls, init_args)`` if ``spec`` is a LightningCLI-style class_path spec.

    Accepts ``{"class_path": "...", "init_args": {...}}`` (Lightning convention).
    Returns ``None`` for plain ``{"kind": ...}`` specs.
    """
    if isinstance(spec, Mapping) and "class_path" in spec:
        return _import_class(spec["class_path"]), dict(spec.get("init_args", {}))
    return None


def _split_spec(spec: Any) -> tuple[str, dict[str, Any]]:
    """Split a ``{"kind": ..., **kwargs}`` mapping into ``(kind, kwargs)``.

    Also accepts a bare string (treated as ``kind`` with no kwargs).
    """
    if isinstance(spec, str):
        return spec, {}
    if isinstance(spec, Mapping):
        kwargs = dict(spec)
        kind = kwargs.pop("kind", None)
        if kind is None:
            raise ValueError(f"Object spec is missing required 'kind' key: {spec!r}")
        return kind, kwargs
    raise TypeError(f"Object spec must be a string or mapping with a 'kind', got {type(spec)}")


def _lookup(kind: str, registry: Mapping[str, type], what: str) -> type:
    if kind not in registry:
        raise KeyError(f"Unknown {what} kind {kind!r}. Available: {sorted(registry)}.")
    return registry[kind]


def _torch_probability_paths() -> dict[str, type]:
    from sc_flow.backends.torch.probability_paths._probability_paths import (
        LinearDiracProbabilityPath,
        LinearGaussianProbabilityPath,
        SchrodingerBridgeProbabilityPath,
        VariancePreservingDiracProbabilityPath,
    )

    return {
        "linear-dirac": LinearDiracProbabilityPath,
        "linear-gaussian": LinearGaussianProbabilityPath,
        "constant-noise-linear-gaussian": LinearGaussianProbabilityPath,
        "schrodinger-bridge-gaussian": SchrodingerBridgeProbabilityPath,
        "variance-preserving-dirac": VariancePreservingDiracProbabilityPath,
    }


def _torch_flow_solvers() -> dict[str, type]:
    from sc_flow.backends.torch.solvers import ODESolver, SDESolver

    return {
        "ode": ODESolver,
        "sde": SDESolver,
    }


_PROBABILITY_PATH_REGISTRIES = {
    "torch": _torch_probability_paths,
}

_FLOW_SOLVER_REGISTRIES = {
    "torch": _torch_flow_solvers,
}


def resolve_probability_path(spec: Any, backend: str) -> Any:
    """Instantiate a probability path from a spec for the given backend."""
    if spec is None:
        return None
    cp = _class_path_spec(spec)
    if cp is not None:
        cls, init_args = cp
        return cls(**init_args)
    if backend not in _PROBABILITY_PATH_REGISTRIES:
        raise NotImplementedError(f"Probability-path resolution not implemented for backend {backend!r}.")
    kind, kwargs = _split_spec(spec)
    cls = _lookup(kind, _PROBABILITY_PATH_REGISTRIES[backend](), "probability path")
    return cls(**kwargs)


def resolve_flow_solver_cls(spec: Any, backend: str) -> tuple[type | None, dict[str, Any]]:
    """Resolve a flow-solver spec to ``(solver_cls, solver_kwargs)``.

    Returns ``(None, {})`` when no solver is configured. ``solver_kwargs`` are
    the extra keyword arguments to pass to prediction/integration (e.g. the
    integration ``scheme`` and ``num_steps``).
    """
    if spec is None:
        return None, {}
    cp = _class_path_spec(spec)
    if cp is not None:
        return cp
    if backend not in _FLOW_SOLVER_REGISTRIES:
        raise NotImplementedError(f"Flow-solver resolution not implemented for backend {backend!r}.")
    kind, kwargs = _split_spec(spec)
    cls = _lookup(kind, _FLOW_SOLVER_REGISTRIES[backend](), "flow solver")
    return cls, kwargs


def _metrics_registry(backend: str) -> dict[str, type]:
    if backend == "torch":
        from sc_flow.backends.torch.metrics import METRICS_REGISTRY

        return dict(METRICS_REGISTRY)
    raise NotImplementedError(f"Metric resolution not implemented for backend {backend!r}.")


def _callback_registry() -> dict[str, type]:
    from sc_flow.trainer._callbacks import MetricsCallback, WandBLogger

    return {"metrics": MetricsCallback, "wandb": WandBLogger}


def _instantiate(spec: Any, registry_fn, what: str) -> Any:
    """Instantiate a ``{kind|class_path: ..., ...}`` spec into an object."""
    cp = _class_path_spec(spec)
    if cp is not None:
        cls, init_args = cp
        return cls(**init_args)
    kind, kwargs = _split_spec(spec)
    cls = _lookup(kind, registry_fn(), what)
    return cls(**kwargs)


def resolve_metrics(specs: Mapping[str, Any], backend: str) -> dict[str, Any]:
    """Instantiate a ``{name: {kind|class_path: ..., ...}}`` metrics mapping."""
    return {
        name: _instantiate(spec, lambda: _metrics_registry(backend), "metric")
        for name, spec in (specs or {}).items()
    }


def resolve_callbacks(trainer_cfg: Any, backend: str) -> list[Any]:
    """Build native callbacks from a :class:`TrainerConfig`.

    ``trainer.metrics`` (if any) are wrapped into a single ``MetricsCallback``;
    ``trainer.callbacks`` entries are resolved via the callback registry or a
    ``class_path``. Returns a list of ``BaseCallback`` instances.
    """
    callbacks: list[Any] = []
    if getattr(trainer_cfg, "metrics", None):
        from sc_flow.trainer._callbacks import MetricsCallback

        metrics = resolve_metrics(trainer_cfg.metrics, backend)
        callbacks.append(MetricsCallback(metrics=metrics, backend=backend, device=trainer_cfg.device))
    for spec in getattr(trainer_cfg, "callbacks", None) or []:
        callbacks.append(_instantiate(spec, _callback_registry, "callback"))
    return callbacks
