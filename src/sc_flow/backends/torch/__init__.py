from __future__ import annotations

from importlib import import_module

__all__ = ["metrics", "nn", "probability_paths", "training"]


def __getattr__(name: str):
    if name in __all__:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(name)
