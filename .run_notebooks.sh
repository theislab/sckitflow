#!/usr/bin/env bash
# Execute every notebook under docs/notebooks/ and report which ones fail.
#
# Notebooks are executed in memory: nothing is written back, so a CI run never
# leaves regenerated outputs in the working tree. Every notebook is attempted
# even after one fails, so a single run reports the full list.
set -euo pipefail

NOTEBOOK_DIR="${1:-docs/notebooks}"

python - "$NOTEBOOK_DIR" <<'PY'
import pathlib
import sys
import time

import nbformat
from jupyter_client.kernelspec import KernelSpec, KernelSpecManager
from jupyter_client.manager import KernelManager
from nbclient import NotebookClient


class CurrentInterpreterKernelSpecManager(KernelSpecManager):
    """Resolve every kernel name to the interpreter running this script.

    The notebooks declare a ``python3`` kernelspec, and a plain name lookup picks
    up whichever ``python3`` kernel happens to be registered user-wide -- which is
    very often some unrelated conda environment, not the one the package was just
    installed into. Pinning to ``sys.executable`` makes the run reproducible.
    """

    def get_kernel_spec(self, kernel_name):
        return KernelSpec(
            argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            display_name=f"Python ({sys.executable})",
            language="python",
        )


notebook_dir = pathlib.Path(sys.argv[1])
notebooks = sorted(notebook_dir.glob("*.ipynb"))
if not notebooks:
    sys.exit(f"no notebooks found under {notebook_dir}")

failed = []
for path in notebooks:
    print(f"::group::{path}", flush=True)
    started = time.time()
    nb = nbformat.read(path, as_version=4)
    km = KernelManager(kernel_name="python3")
    km.kernel_spec_manager = CurrentInterpreterKernelSpecManager()
    client = NotebookClient(
        nb,
        timeout=1800,
        km=km,
        # Run with the notebook's own directory as cwd so relative paths resolve.
        resources={"metadata": {"path": str(path.parent)}},
    )
    try:
        client.execute()
    except Exception as exc:  # report every failure, not just the first
        failed.append(path)
        print(f"FAIL {path} ({time.time() - started:.1f}s)\n{exc}", flush=True)
    else:
        print(f"PASS {path} ({time.time() - started:.1f}s)", flush=True)
    print("::endgroup::", flush=True)

if failed:
    sys.exit("failed notebooks:\n" + "\n".join(f"  {p}" for p in failed))
print(f"all {len(notebooks)} notebooks executed successfully")
PY
