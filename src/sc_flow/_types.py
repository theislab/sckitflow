from collections.abc import Callable, Mapping
from typing import Any, Literal

import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

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

CouplingSpaceReps = Literal[
    "src_coupling_lin",
    "tgt_coupling_lin",
    "src_coupling_quad",
    "tgt_coupling_quad",
]

TargetCovariatesEncoding = Literal["label", "one-hot", "identity"]

MappedArray = dict[str, np.ndarray]

ArrayTransformation = Callable[[np.ndarray], np.ndarray]
TargetCovariatesEncoder = ArrayTransformation | LabelEncoder | OneHotEncoder
MappedCovariatesEncoder = Mapping[str, TargetCovariatesEncoder]
