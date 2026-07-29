"""Serializable group encoders.

Each encoder is a frozen dataclass of plain scalars -- no callables -- so it pickles
trivially and compares by value. ``build`` fits and returns a transformer exposing
``transform`` (and, for the functional encoders, ``inverse_transform``). This replaces
the string encoder ids + raw ``groups_encoding_transform_fn`` callables, which could not
be serialized inside a ``DataManager``.

The set is closed for now; add a new encoder as a ``GroupEncoder`` subclass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder

from sc_flow._types import TargetCovariatesEncoderCls

__all__ = ["GroupEncoder", "GroupEncoderContext", "Identity", "Label", "OneHot", "Affine", "Log1p"]


@dataclass(frozen=True)
class GroupEncoderContext:
    """Fit-time inputs for a group encoder -- deliberately NOT part of the serialized config."""

    data: np.ndarray


class GroupEncoder:
    """Base for serializable group encoders. ``build(context)`` returns a fitted transformer."""

    def build(self, context: GroupEncoderContext) -> TargetCovariatesEncoderCls:
        raise NotImplementedError


@dataclass(frozen=True)
class Label(GroupEncoder):
    def build(self, context: GroupEncoderContext) -> LabelEncoder:
        return LabelEncoder().fit(np.asarray(context.data).reshape(-1))


@dataclass(frozen=True)
class OneHot(GroupEncoder):
    def build(self, context: GroupEncoderContext) -> OneHotEncoder:
        return OneHotEncoder().fit(np.asarray(context.data).reshape(-1, 1))


@dataclass(frozen=True)
class Identity(GroupEncoder):
    def build(self, context: GroupEncoderContext) -> FunctionTransformer:
        return FunctionTransformer(check_inverse=False).fit(context.data)


@dataclass(frozen=True)
class Log1p(GroupEncoder):
    def build(self, context: GroupEncoderContext) -> FunctionTransformer:
        return FunctionTransformer(func=np.log1p, inverse_func=np.expm1, check_inverse=False).fit(context.data)


@dataclass(frozen=True)
class Affine(GroupEncoder):
    scale: float = 1.0
    shift: float = 0.0

    def build(self, context: GroupEncoderContext) -> FunctionTransformer:
        s, b = self.scale, self.shift
        return FunctionTransformer(
            func=lambda x: x * s + b,
            inverse_func=lambda x: (x - b) / s,
            check_inverse=False,
        ).fit(context.data)
