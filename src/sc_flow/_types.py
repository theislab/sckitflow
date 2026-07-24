from typing import Any, Literal

#: A layer-config mapping (kwargs for one nn layer).
LayersDict = dict[str, Any]

ProbabilityPathId = Literal[
    "constant-noise-linear-gaussian",
    "cnlg-pp",
    "schrodinger-bridge-gaussian",
    "sbg-pp",
    "variance-preserving-dirac",
    "vpd-pp",
    "linear-dirac",
    "ld-pp",
]
TimeFeaturesId = Literal["sinusoidal", "log-sinusoidal"]
