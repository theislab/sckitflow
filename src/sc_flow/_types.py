from typing import Any, Literal, TypeAlias

import numpy as np
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder

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

TargetCovariatesEncodingId = Literal["label", "one-hot", "functional"]

MappedArray: TypeAlias = dict[str, np.ndarray]

TargetCovariatesEncoderCls = FunctionTransformer | LabelEncoder | OneHotEncoder
