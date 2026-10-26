# sckitflow

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/theislab/sckitflow/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/sckitflow

Flow Matching tool-box for Single Cell Data.

## Getting started

Please refer to the [documentation][],
in particular, the [API documentation][].

## Installation

You need to have Python 3.12 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

There are several alternative options to install sckitflow:

<!--
1) Install the latest release of `sckitflow` from [PyPI][]:

```bash
pip install sckitflow
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/theislab/sckitflow.git@main
```

For GPU-resident streaming (`preload_to_gpu=True`), add the cupy extra matching the CUDA major of your torch build:

| Extra | What it adds |
| --- | --- |
| `cupy-cuda12` | cupy for CUDA 12, with its CUDA toolkit wheels |
| `cupy-cuda13` | cupy for CUDA 13, with its CUDA toolkit wheels |

```bash
pip install "sckitflow[cupy-cuda13] @ git+https://github.com/theislab/sckitflow.git@main"
```

In a uv-managed checkout, `uv sync --extra cu130` (or `cu126`, `cpu`) picks torch from the matching PyTorch index together with the cupy of the same CUDA major.

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Citation

> t.b.a

[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/theislab/sckitflow/issues
[tests]: https://github.com/theislab/sckitflow/actions/workflows/test.yaml
[documentation]: https://sckitflow.readthedocs.io
[changelog]: https://sckitflow.readthedocs.io/en/latest/changelog.html
[api documentation]: https://sckitflow.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/sckitflow
