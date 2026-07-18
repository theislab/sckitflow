"""Lazy import of the optional heavy backends with a clear ``pip install`` hint.

``import sc_flow`` and ``import sc_flow.data`` are pure-Python and must not pull torch / jax /
lightning (they are declared as *extras*, not core deps). The model + training subsystems import
them lazily at use; when the extra is not installed that would surface as a bare
``ModuleNotFoundError: No module named 'torch'`` deep in a traceback. :func:`require` turns that into a
single actionable line naming the extra to install. This module itself imports nothing heavy.
"""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType

__all__ = ["require", "EXTRA_FOR"]

# Root module -> the ``sc-flow-tools[<extra>]`` that provides it (see pyproject ``optional-dependencies``).
EXTRA_FOR: dict[str, str] = {
    "torch": "torch",
    "torchdiffeq": "torch",
    "torchsde": "torch",
    "torchmetrics": "torch",
    "ot": "torch",  # POT
    "lightning": "lightning",
    "pytorch_lightning": "lightning",
    "jax": "jax",
    "jaxlib": "jax",
    "ott": "jax",
    "flax": "jax",
    "diffrax": "jax",
}


def _root_absent(root: str) -> bool:
    """True if top-level ``root`` cannot be found (missing, or a finder hard-blocks it).

    In a real bare env ``find_spec`` returns ``None`` for an uninstalled top-level module; a meta-path
    finder that raises for a blocked root (our bare-env test harness) is likewise treated as absent.
    """
    try:
        return importlib.util.find_spec(root) is None
    except ModuleNotFoundError:
        return True


def require(module: str) -> ModuleType:
    """Import ``module`` (dotted path), re-raising a missing *optional backend* with an install hint.

    The translation keys on the module that is actually *missing* (``e.name``), not on the requested
    path — so ``require("sc_flow._model")`` on a bare env (where ``sc_flow._model`` internally does
    ``import torch``) still reports the torch extra. Only the heavy roots in :data:`EXTRA_FOR` are
    reinterpreted; an unrelated ``ModuleNotFoundError`` (a real bug inside an installed backend)
    propagates unchanged.
    """
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as e:
        missing_root = (e.name or "").split(".")[0]
        extra = EXTRA_FOR.get(missing_root)
        # Only translate when the backend ROOT is genuinely absent. A missing *sub*module of an installed
        # backend (e.name is then the full dotted path, e.g. a real "torch.optimm" typo) must propagate its
        # true traceback — not get mis-reported as "install torch".
        if extra is not None and _root_absent(missing_root):
            raise ModuleNotFoundError(
                f"{module!r} needs the optional {missing_root!r} backend, which is not installed. "
                f"Install it with:  pip install 'sc-flow-tools[{extra}]'  "
                f"(or 'sc-flow-tools[all]' for every backend; add '[cuda]' on a GPU node for jax CUDA)."
            ) from e
        raise
