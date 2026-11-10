from typing import Any, Literal

from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder

BackendId = Literal["torch", "jax"]

ActivationId = Literal["relu", "leaky_relu", "elu", "gelu", "silu", "tanh", "sigmoid", "softplus"]

LayersDict = dict[str, Any]
NestedLayersDict = dict[str, LayersDict]


TargetCovariatesEncodingId = Literal["label", "one-hot", "functional"]

TargetCovariatesEncoderCls = FunctionTransformer | LabelEncoder | OneHotEncoder
