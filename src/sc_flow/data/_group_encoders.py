"""Serializable group encoders built on :class:`scfit.registry.Component`.

Each encoder is a frozen dataclass of plain scalars -- no callables -- so it pickles trivially, compares by
value, AND exports a portable ``{type, version, config}`` spec via ``.to_spec()`` (round-tripped with
``GroupEncoder.from_spec``). ``build`` fits and returns a transformer exposing ``transform`` (and, for the
functional encoders, ``inverse_transform``). This replaces the string encoder ids + raw
``groups_encoding_transform_fn`` callables, which could not be serialized inside a ``DataManager``.

The stateful encoders (:class:`Label`, :class:`OneHot`) learn their vocabulary from data by default, but
that vocabulary can be *pinned* in the config (``classes=`` / ``categories=``). A pinned encoder rebuilds
the exact same mapping regardless of the data ``build`` later sees -- so a serialized ``DataManager`` cannot
silently drift its categorical encoding at predict time -- while the config stays portable JSON (no pickled
sklearn estimator). Leave the field ``None`` to fit from data as before.

``GroupEncoder`` is the abstract family base (a ``Component`` with no ``type_id``); each concrete encoder
opts in by passing ``type_id=`` in its class header, which auto-registers it. Add a new encoder the same
way: subclass ``GroupEncoder``, give it a ``type_id``, own its ``build``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scfit.registry import Component
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder

from sc_flow._types import TargetCovariatesEncoderCls

__all__ = ["GroupEncoder", "GroupEncoderContext", "Identity", "Label", "OneHot", "Affine", "Log1p"]


@dataclass(frozen=True)
class GroupEncoderContext:
    """Fit-time inputs for a group encoder -- deliberately NOT part of the serialized config."""

    data: np.ndarray


class GroupEncoder(Component):
    """Abstract family base for serializable group encoders (a ``Component`` with no ``type_id``).

    ``build(context)`` returns a fitted transformer; concrete subclasses register via ``type_id=``.
    """

    def build(self, context: GroupEncoderContext) -> TargetCovariatesEncoderCls:
        raise NotImplementedError


@dataclass(frozen=True)
class Label(GroupEncoder, type_id="group_encoder.label"):
    """Integer-code encoder. ``classes`` pins the vocabulary; ``None`` fits it from data.

    Note that ``LabelEncoder`` sorts its classes, so the code assigned to a label is the position of that
    label in the sorted vocabulary either way -- pinning only fixes *which* labels exist.
    """

    classes: tuple[str, ...] | None = None

    def build(self, context: GroupEncoderContext) -> LabelEncoder:
        values = context.data if self.classes is None else self.classes
        return LabelEncoder().fit(np.asarray(values).reshape(-1))


@dataclass(frozen=True)
class OneHot(GroupEncoder, type_id="group_encoder.one_hot"):
    """One-hot encoder. ``categories`` pins the column order/vocabulary; ``None`` fits it from data."""

    categories: tuple[str, ...] | None = None

    def build(self, context: GroupEncoderContext) -> OneHotEncoder:
        if self.categories is None:
            return OneHotEncoder().fit(np.asarray(context.data).reshape(-1, 1))
        cats = list(self.categories)
        return OneHotEncoder(categories=[cats]).fit(np.asarray(cats).reshape(-1, 1))


@dataclass(frozen=True)
class Identity(GroupEncoder, type_id="group_encoder.identity"):
    def build(self, context: GroupEncoderContext) -> FunctionTransformer:
        return FunctionTransformer(check_inverse=False).fit(context.data)


@dataclass(frozen=True)
class Log1p(GroupEncoder, type_id="group_encoder.log1p"):
    def build(self, context: GroupEncoderContext) -> FunctionTransformer:
        return FunctionTransformer(func=np.log1p, inverse_func=np.expm1, check_inverse=False).fit(context.data)


@dataclass(frozen=True)
class Affine(GroupEncoder, type_id="group_encoder.affine"):
    scale: float = 1.0
    shift: float = 0.0

    def build(self, context: GroupEncoderContext) -> FunctionTransformer:
        s, b = self.scale, self.shift
        return FunctionTransformer(
            func=lambda x: x * s + b,
            inverse_func=lambda x: (x - b) / s,
            check_inverse=False,
        ).fit(context.data)
