from collections.abc import Mapping
from typing import Any, Literal, TypeAlias, TypeVar

import numpy as np
import pandas as pd
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

CouplingSpaceReps = Literal[
    "src_coupling_lin",
    "tgt_coupling_lin",
    "src_coupling_quad",
    "tgt_coupling_quad",
]

TargetCovariatesEncodingId = Literal["label", "one-hot", "identity"]

MappedArray: TypeAlias = dict[str, np.ndarray]

TargetCovariatesEncoderCls = FunctionTransformer | LabelEncoder | OneHotEncoder


MappedLevelIndex: TypeAlias = Mapping[tuple[Any, ...], pd.MultiIndex]
NamedMappedLevelIndex: TypeAlias = Mapping[str, str | MappedLevelIndex]

T = TypeVar("T", "MappedLevelIndex", "NamedMappedLevelIndex")
NestedMappedLevelIndex: TypeAlias = Mapping[
    tuple[Any, ...],
    "NestedMappedLevelIndex | T",
]
NamedNestedMappedLevelIndex: TypeAlias = Mapping[
    str,
    "NamedNestedMappedLevelIndex | T",
]
