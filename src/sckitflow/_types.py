import abc
from collections.abc import Collection
from typing import Any, Literal

from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder

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

TargetCovariatesEncoderCls = FunctionTransformer | LabelEncoder | OneHotEncoder

GENOTDataMatchFn = Any  # TODO

TensorLike = Any  # TODO


class PredictionData:
    X: Any
    raw_samples: Any | None = None
    traj: Any | None = None

    @classmethod
    @abc.abstractmethod
    def concatenate(cls, preds: Collection["PredictionData"]) -> "PredictionData": ...
