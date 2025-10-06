from typing import Any, Literal

BackendId = Literal["torch", "jax"]
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
TimeFeaturesId = Literal["ott-jax", "torch-cfm"]

ConditioningLayersId = Literal["concat", "resnet1d"]

LayersDict = dict[str, Any]
NestedLayersDict = dict[str, LayersDict]
