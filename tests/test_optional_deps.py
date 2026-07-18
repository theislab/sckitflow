"""Lazy / optional heavy-dep boundary: ``import sc_flow`` and ``sc_flow.data`` must work without
torch / jax / lightning installed, and touching a heavy subsystem must raise a clear install hint.

Each case runs in a fresh subprocess with a meta-path finder that makes the heavy backends look
absent (their ``ModuleNotFoundError`` carries ``.name`` like CPython's), so the real import graph is
exercised without uninstalling anything from the test env.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

# Roots the blocker hides — the optional backends declared as pyproject extras.
_BLOCK = ("torch", "jax", "jaxlib", "lightning", "pytorch_lightning", "ott", "flax", "diffrax",
          "torchdiffeq", "torchsde", "torchmetrics", "ot")

_BLOCKER = f"""
import sys
BLOCK = {_BLOCK!r}
class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in BLOCK:
            raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)
        return None
for _m in list(sys.modules):
    if _m.split('.')[0] in BLOCK:
        del sys.modules[_m]
sys.meta_path.insert(0, _Blocker())
"""


def _run(body: str) -> subprocess.CompletedProcess:
    """Run ``body`` in a subprocess with the heavy backends blocked; return the completed process."""
    return subprocess.run(
        [sys.executable, "-c", _BLOCKER + textwrap.dedent(body)],
        capture_output=True, text=True,
    )


def test_import_sc_flow_without_heavy_backends():
    r = _run("import sc_flow; print('ok')")
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout


def test_import_sc_flow_data_without_heavy_backends():
    r = _run(
        """
        import sc_flow.data as d
        d.FlowSpec; d.compile_obs
        import sc_flow.data.schemas, sc_flow.data.containers, sc_flow.data.sim
        print('ok')
        """
    )
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout


def test_flowmatching_access_raises_clear_install_hint():
    r = _run(
        """
        import sc_flow
        try:
            sc_flow.FlowMatching
        except ModuleNotFoundError as e:
            print("MSG:", e)
        else:
            print("NO_ERROR")
        """
    )
    assert r.returncode == 0, r.stderr
    assert "sc-flow-tools[torch]" in r.stdout
    assert "NO_ERROR" not in r.stdout


def test_lazy_submodules_import_without_heavy_backends():
    """The lazy submodules are internally lazy too: their ``__init__`` pulls no heavy backend."""
    r = _run(
        """
        import sc_flow
        for name in ("backends", "methods", "trainer", "dataset"):
            getattr(sc_flow, name)
        print("ok")
        """
    )
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout


def test_require_translates_only_optional_backends():
    """require() re-raises an unrelated missing module untouched (no false install hint)."""
    r = _run(
        """
        from sc_flow._optional import require
        try:
            require("sc_flow._definitely_missing_module_xyz")
        except ModuleNotFoundError as e:
            print("MSG:", e)
        """
    )
    assert r.returncode == 0, r.stderr
    assert "sc-flow-tools[" not in r.stdout
    assert "_definitely_missing_module_xyz" in r.stdout
