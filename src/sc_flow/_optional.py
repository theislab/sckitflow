
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
    try:
        return importlib.util.find_spec(root) is None
    except ModuleNotFoundError:
        return True


def require(module: str) -> ModuleType:
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
