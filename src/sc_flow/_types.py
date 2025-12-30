from collections.abc import Mapping
from dataclasses import dataclass
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


_T = TypeVar("_T", bound="MappedLevelIndex")


@dataclass
class MappedLevelIndex:
    mapping: Mapping[tuple[Any, ...], "_T | pd.MultiIndex"]

    @property
    def is_leaf(self) -> bool:
        return all(isinstance(v, pd.MultiIndex) for v in self.mapping.values())
