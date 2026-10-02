"""Serializable group encoders built on :class:`scfit.registry.Component`.

Each encoder is a frozen model of plain scalars -- no callables -- so it pickles trivially, compares by
value, AND exports a portable ``{type, version, **fields}`` spec via ``.to_spec()`` (round-tripped with
``GroupEncoderConfig.from_spec``). ``build`` fits and returns a transformer exposing ``transform`` (and, for the
functional encoders, ``inverse_transform``). This replaces the string encoder ids + raw
``groups_encoding_transform_fn`` callables, which could not be serialized inside a ``DataManager``.

``GroupEncoderConfig`` is the family base; each config subclasses it, is registered with
``@component(type_id, builds=X)`` and named ``XConfig`` after the transformer ``X`` it builds. Add a new encoder the same way.

The stateful encoders (:class:`LabelEncoderConfig`, :class:`OneHotEncoderConfig`) accept an optional **pinned vocabulary** so a
serialized config rebuilds the *exact* same mapping instead of re-deriving one from whatever data ``build``
happens to see. Unknown categories always **raise** -- never silently ignored -- so train/predict skew
fails loudly rather than producing quietly wrong codes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scfit.registry import Component, component
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder

from sckitflow._types import TargetCovariatesEncoderCls

__all__ = [
    "GroupEncoderConfig",
    "GroupEncoderContext",
    "GroupEncoderId",
    "LabelEncoderConfig",
    "OneHotEncoderConfig",
    "IdentityTransformer",
    "IdentityTransformerConfig",
    "Log1pTransformer",
    "Log1pTransformerConfig",
    "AffineTransformer",
    "AffineTransformerConfig",
    "as_group_encoder",
]

#: String ids accepted at the public interfaces as a shorthand for the parameter-free encoders.
GroupEncoderId = Literal["label", "one-hot"]


@dataclass(frozen=True)
class GroupEncoderContext:
    """Fit-time inputs for a group encoder -- deliberately NOT part of the serialized config."""

    data: np.ndarray


class GroupEncoderConfig(Component):
    """Family base for serializable group encoders. ``build(context)`` returns the fitted transformer."""

    def build(self, context: GroupEncoderContext) -> TargetCovariatesEncoderCls:
        raise NotImplementedError


@component("group_encoder.label", builds=LabelEncoder)
class LabelEncoderConfig(GroupEncoderConfig):
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


@component("group_encoder.one_hot", builds=OneHotEncoder)
class OneHotEncoderConfig(GroupEncoderConfig):
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


class IdentityTransformer(FunctionTransformer):
    """Passes values through unchanged."""

    def __init__(self) -> None:
        super().__init__(check_inverse=False)


class Log1pTransformer(FunctionTransformer):
    """``log1p``, inverted by ``expm1``."""

    def __init__(self) -> None:
        super().__init__(func=np.log1p, inverse_func=np.expm1, check_inverse=False)


class AffineTransformer(FunctionTransformer):
    """``x * scale + shift``, inverted by ``(x - shift) / scale``."""

    def __init__(self, scale: float = 1.0, shift: float = 0.0) -> None:
        self.scale, self.shift = scale, shift
        super().__init__(func=self._forward, inverse_func=self._inverse, check_inverse=False)

    def _forward(self, x: np.ndarray) -> np.ndarray:
        return x * self.scale + self.shift

    def _inverse(self, x: np.ndarray) -> np.ndarray:
        return (x - self.shift) / self.scale


@component("group_encoder.identity", builds=IdentityTransformer)
class IdentityTransformerConfig(GroupEncoderConfig):
    """Passes the column through unchanged."""

    def build(self, context: GroupEncoderContext) -> IdentityTransformer:
        return IdentityTransformer().fit(context.data)


@component("group_encoder.log1p", builds=Log1pTransformer)
class Log1pTransformerConfig(GroupEncoderConfig):
    """Applies ``log1p`` (inverse ``expm1``) to a continuous column."""

    def build(self, context: GroupEncoderContext) -> Log1pTransformer:
        return Log1pTransformer().fit(context.data)


@component("group_encoder.affine", builds=AffineTransformer)
class AffineTransformerConfig(GroupEncoderConfig):
    """Scales and shifts a continuous column (``x * scale + shift``).

    :param scale: Multiplicative factor. Defaults to ``1.0``.
    :type scale: class: `float`

    :param shift: Additive offset. Defaults to ``0.0``.
    :type shift: class: `float`
    """

    scale: float = 1.0
    shift: float = 0.0

    def build(self, context: GroupEncoderContext) -> AffineTransformer:
        return AffineTransformer(scale=self.scale, shift=self.shift).fit(context.data)


#: The string ids, mapped to their component equivalent. Deliberately only the parameter-free encoders:
#: the legacy ``"functional"`` id carried its transform in the separate ``groups_encoding_transform_fn``
#: callables, so with those gone it has no meaning as a string -- pass :class:`IdentityTransformerConfig`,
#: :class:`Log1pTransformerConfig` or :class:`AffineTransformerConfig` explicitly instead.
_ENCODER_BY_ID: dict[str, type[GroupEncoderConfig]] = {
    "label": LabelEncoderConfig,
    "one-hot": OneHotEncoderConfig,
}


def as_group_encoder(value: GroupEncoderConfig | GroupEncoderId) -> GroupEncoderConfig:
    """Coerces a string encoder id into a :class:`GroupEncoderConfig`, passing instances through.

    Strings are a convenience accepted only at the public interfaces (``DataManager`` /
    ``GroupsDataSchema``); everything downstream stores components. Only the parameter-free encoders have
    string ids -- reach for the instance (``AffineTransformerConfig(scale=2.0)``, ``OneHotEncoderConfig(categories=(...))``) when you need
    parameters or a pinned vocabulary.

    :param value: A :class:`GroupEncoderConfig` instance, or one of ``"label"`` / ``"one-hot"``.
    :type value: class: `GroupEncoderConfig | GroupEncoderId`
    """
    if isinstance(value, GroupEncoderConfig):
        return value
    try:
        encoder_cls = _ENCODER_BY_ID[value]
    except (KeyError, TypeError):
        msg = f"Group encoder {value!r} not available. Pass a GroupEncoderConfig instance or one of {sorted(_ENCODER_BY_ID)}."
        raise ValueError(msg) from None
    return encoder_cls()
