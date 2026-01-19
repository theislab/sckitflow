from types import ModuleType

from sc_flow._constants import DEFAULT_BACKEND
from sc_flow._types import BackendId

BACKEND = DEFAULT_BACKEND


TORCH_IMPORT_FAILED = False
JAX_IMPORT_FAILED = False
TQDM_IMPORT_FAILED = False
NUMBA_IMPORT_FAILED = False


def set_backend(backend: BackendId) -> None:
    global BACKEND
    if backend not in ["torch", "jax"]:
        raise_runtime_error_on_backend_not_supported(backend)
    BACKEND = backend


def set_torch_import_failed(failed: bool) -> None:
    global TORCH_IMPORT_FAILED
    TORCH_IMPORT_FAILED = failed


def set_jax_import_failed(failed: bool) -> None:
    global JAX_IMPORT_FAILED
    JAX_IMPORT_FAILED = failed


def set_tqdm_import_failed(failed: bool) -> None:
    global TQDM_IMPORT_FAILED
    TQDM_IMPORT_FAILED = failed


def set_numba_import_failed(failed: bool) -> None:
    global NUMBA_IMPORT_FAILED
    NUMBA_IMPORT_FAILED = failed


def raise_runtime_error_on_backend_not_supported(backend: str) -> None:
    msg = f'{backend} not supported, possible choices are `["torch", "jax"]`'
    raise RuntimeError(msg)


def raise_runtime_error_on_backend_failed_import() -> None:
    global BACKEND, TORCH_IMPORT_FAILED, JAX_IMPORT_FAILED
    if BACKEND == "torch" and TORCH_IMPORT_FAILED:
        msg = "Failed to import torch backend."
        raise RuntimeError(msg)
    if BACKEND == "jax" and JAX_IMPORT_FAILED:
        msg = "Failed to import jax backend."
        raise RuntimeError(msg)


def attempt_tqdm_import() -> ModuleType | None:
    try:
        from tqdm.auto import tqdm
    except ImportError:
        set_tqdm_import_failed(True)
        tqdm = None
    return tqdm


def attempt_numba_import() -> ModuleType | None:
    try:
        import numba as nb
    except ImportError:
        set_numba_import_failed(True)
        nb = None
    return nb
