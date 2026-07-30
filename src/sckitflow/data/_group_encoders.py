"""Serializable group encoders built on :class:`scfit.registry.Component`.

Each encoder is a frozen dataclass of plain scalars -- no callables -- so it pickles trivially, compares by
value, AND exports a portable ``{type, version, config}`` spec via ``.to_spec()`` (round-tripped with
``GroupEncoder.from_spec``). ``build`` fits and returns a transformer exposing ``transform`` (and, for the
functional encoders, ``inverse_transform``). This replaces the string encoder ids + raw
``groups_encoding_transform_fn`` callables, which could not be serialized inside a ``DataManager``.

``GroupEncoder`` is the abstract family base (a ``Component`` with no ``type_id``); each concrete encoder
opts in by passing ``type_id=`` in its class header, which auto-registers it. Add a new encoder the same
way: subclass ``GroupEncoder``, give it a ``type_id``, own its ``build``.

The stateful encoders (:class:`Label`, :class:`OneHot`) accept an optional **pinned vocabulary** so a
serialized config rebuilds the *exact* same mapping instead of re-deriving one from whatever data ``build``
happens to see. Unknown categories always **raise** -- never silently ignored -- so train/predict skew
fails loudly rather than producing quietly wrong codes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scfit.registry import Component
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder

from sckitflow._types import TargetCovariatesEncoderCls

__all__ = [
    "GroupEncoder",
    "GroupEncoderContext",
    "GroupEncoderId",
    "Identity",
    "Label",
    "OneHot",
    "Affine",
    "Log1p",
    "as_group_encoder",
]

#: String ids accepted at the public interfaces as a shorthand for the parameter-free encoders.
GroupEncoderId = Literal["label", "one-hot"]


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
    """Integer-codes a categorical column.

    :param classes: Pinned vocabulary. ``None`` derives it from the data at fit time (order is
        ``np.unique``, i.e. sorted). When set, the mapping is exactly ``classes`` in the given order, so a
        round-tripped config reproduces identical codes. ``transform`` raises on any value not listed.
    :type classes: class: `tuple[str, ...] | None`
    """

    classes: tuple[str, ...] | None = None

    def build(self, context: GroupEncoderContext) -> LabelEncoder:
        # Fitting on the vocabulary itself pins the mapping; unseen labels raise in transform.
        values = np.asarray(context.data).reshape(-1) if self.classes is None else np.asarray(self.classes)
        return LabelEncoder().fit(values)


@dataclass(frozen=True)
class OneHot(GroupEncoder, type_id="group_encoder.one_hot"):
    """One-hot encodes a categorical column.

    :param categories: Pinned vocabulary. ``None`` derives it from the data at fit time. When set, it fixes
        the column order, so a round-tripped config reproduces identical columns -- do not reorder it. Fit
        raises if the data holds a value outside it, and ``transform`` raises on unknown categories.
    :type categories: class: `tuple[str, ...] | None`
    """

    categories: tuple[str, ...] | None = None

    def build(self, context: GroupEncoderContext) -> OneHotEncoder:
        # handle_unknown="error" is sklearn's default; set explicitly because pinning a vocabulary makes
        # "what happens to an unlisted category" a load-bearing decision -- it must fail loudly.
        categories = "auto" if self.categories is None else [list(self.categories)]
        return OneHotEncoder(categories=categories, handle_unknown="error").fit(np.asarray(context.data).reshape(-1, 1))


@dataclass(frozen=True)
class Identity(GroupEncoder, type_id="group_encoder.identity"):
    """Passes the column through unchanged."""

    def build(self, context: GroupEncoderContext) -> FunctionTransformer:
        return FunctionTransformer(check_inverse=False).fit(context.data)


@dataclass(frozen=True)
class Log1p(GroupEncoder, type_id="group_encoder.log1p"):
    """Applies ``log1p`` (inverse ``expm1``) to a continuous column."""

    def build(self, context: GroupEncoderContext) -> FunctionTransformer:
        return FunctionTransformer(func=np.log1p, inverse_func=np.expm1, check_inverse=False).fit(context.data)


@dataclass(frozen=True)
class Affine(GroupEncoder, type_id="group_encoder.affine"):
    """Scales and shifts a continuous column (``x * scale + shift``).

    :param scale: Multiplicative factor. Defaults to ``1.0``.
    :type scale: class: `float`

    :param shift: Additive offset. Defaults to ``0.0``.
    :type shift: class: `float`
    """

    scale: float = 1.0
    shift: float = 0.0

    def build(self, context: GroupEncoderContext) -> FunctionTransformer:
        s, b = self.scale, self.shift
        return FunctionTransformer(
            func=lambda x: x * s + b,
            inverse_func=lambda x: (x - b) / s,
            check_inverse=False,
        ).fit(context.data)


#: The string ids, mapped to their component equivalent. Deliberately only the parameter-free encoders:
#: the legacy ``"functional"`` id carried its transform in the separate ``groups_encoding_transform_fn``
#: callables, so with those gone it has no meaning as a string -- pass :class:`Identity`, :class:`Log1p`
#: or :class:`Affine` explicitly instead.
_ENCODER_BY_ID: dict[str, type[GroupEncoder]] = {
    "label": Label,
    "one-hot": OneHot,
}


def as_group_encoder(value: GroupEncoder | GroupEncoderId) -> GroupEncoder:
    """Coerces a string encoder id into a :class:`GroupEncoder`, passing instances through.

    Strings are a convenience accepted only at the public interfaces (``DataManager`` /
    ``GroupsDataSchema``); everything downstream stores components. Only the parameter-free encoders have
    string ids -- reach for the instance (``Affine(scale=2.0)``, ``OneHot(categories=(...))``) when you need
    parameters or a pinned vocabulary.

    :param value: A :class:`GroupEncoder` instance, or one of ``"label"`` / ``"one-hot"``.
    :type value: class: `GroupEncoder | GroupEncoderId`
    """
    if isinstance(value, GroupEncoder):
        return value
    try:
        encoder_cls = _ENCODER_BY_ID[value]
    except (KeyError, TypeError):
        msg = f"Group encoder {value!r} not available. Pass a GroupEncoder instance or one of {sorted(_ENCODER_BY_ID)}."
        raise ValueError(msg) from None
    return encoder_cls()
