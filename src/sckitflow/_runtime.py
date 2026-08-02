from types import ModuleType

TQDM_IMPORT_FAILED = False


def set_tqdm_import_failed(failed: bool) -> None:
    global TQDM_IMPORT_FAILED
    TQDM_IMPORT_FAILED = failed


def attempt_tqdm_import() -> ModuleType | None:
    try:
        from tqdm.auto import tqdm
    except ImportError:
        set_tqdm_import_failed(True)
        tqdm = None
    return tqdm
