"""Per-realm condition encoders: the model-side ``index|feature -> embedding`` map.

A *realm* is one condition covariate group (a set of combinatorial slots, e.g. ``drug_1``/``drug_2``). The
dataloader side (:func:`~sc_flow.data._compile_obs`) is a pure lookup — it emits, per realm, either an
integer **index** (categorical) or a looked-up **feature vector**. The *encoding* lives here, model-side and
trainable, so a condition encoder can be a learned embedding / linear projection rather than a frozen
one-hot, and trains jointly with the flow.

This is a closed-but-extensible discriminated-spec family on the shared :class:`~scfit.ComponentRegistry`
(same machinery as pooling/combiner). Built-ins: ``sc_flow.embedding`` (learned), ``sc_flow.onehot``
(fixed), ``sc_flow.feature_mlp`` (projection of a looked-up vector). New kinds — e.g. an LLM/text encoder
over SMILES or protein sequences — are added by *registering* another config, with no change here; the
data side is untouched because ``embedding``/``onehot``/``llm`` all consume the same integer **index**,
only ``feature_mlp`` consumes a vector.

Kept in ``sc_flow.flow`` (not the generic ``scfit`` base): the only consumer is the perturbation
:class:`~sc_flow.flow._set_encoder.SetEncoder`, ``realm`` is perturbation vocabulary, and open members
(the LLM encoder) carry heavy deps that must not enter the lean base. Promote to a generic input-encoder
family only if a non-perturbation consumer appears.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from scfit._component import ComponentRegistry, ComponentSpec, JsonValue
from scfit.nn._utils import init_module_from_dict

__all__ = [
    "RealmContext",
    "RealmEncoderSpec",
    "OneHot",
    "IndexEmbedding",
    "REALM_ENCODER_REGISTRY",
    "validate_realm_encoder_spec",
    "build_realm_encoder",
    "realm_output_dim",
]

#: A realm-encoder spec is a :class:`~scfit.ComponentSpec` — the slot (a condition realm) fixes the family.
RealmEncoderSpec = ComponentSpec


@dataclass(frozen=True)
class RealmContext:
    """Build context for a realm encoder.

    Empty today — every built-in is fully sized from its own config (``num_categories`` / ``input_dim`` are
    resolved at fit-time and baked into the spec for portable reload). Kept as a named type for parity with
    the pooling/combiner build contexts and as the extension point for any future derived input.
    """


# These encoders consume an integer index and do NO dtype/device coercion — that is the compute boundary's
# job (``Objective.compute_loss`` / ``Predictor.predict`` place + type the batch), and the dataloader
# preserves the int dtype the index carries. The modules assume a ``long`` index on the right device.
class OneHot(torch.nn.Module):
    """Fixed, weightless ``index -> one-hot`` encoder. ``output_dim == num_categories``.

    The frozen special case of a learned embedding: swapping ``sc_flow.onehot`` for ``sc_flow.embedding``
    changes only whether the ``index -> vector`` table has learnable weights. Expects a ``long`` index.
    """

    def __init__(self, num_categories: int) -> None:
        super().__init__()
        if num_categories <= 0:
            raise ValueError(f"num_categories must be positive, found {num_categories}.")
        self.num_categories = num_categories

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(idx, num_classes=self.num_categories).to(torch.get_default_dtype())


class IndexEmbedding(torch.nn.Module):
    """Learned ``index -> vector`` table (``nn.Embedding``). Expects a ``long`` index."""

    def __init__(self, num_categories: int, output_dim: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(num_categories, output_dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.embedding(idx)


REALM_ENCODER_REGISTRY: ComponentRegistry[RealmContext] = ComponentRegistry("realm_encoder")


@REALM_ENCODER_REGISTRY.register("sc_flow.embedding")
@dataclass(frozen=True)
class EmbeddingRealmConfig:
    """Learned ``index -> vector`` table (``nn.Embedding``). Trains jointly with the flow."""

    num_categories: int
    output_dim: int

    def __post_init__(self) -> None:
        if self.num_categories <= 0 or self.output_dim <= 0:
            raise ValueError(
                f"embedding realm needs positive num_categories/output_dim, "
                f"found num_categories={self.num_categories}, output_dim={self.output_dim}."
            )

    def build(self, context: RealmContext) -> torch.nn.Module:
        return IndexEmbedding(self.num_categories, self.output_dim)


@REALM_ENCODER_REGISTRY.register("sc_flow.onehot")
@dataclass(frozen=True)
class OneHotRealmConfig:
    """Fixed ``index -> one-hot`` encoder; ``output_dim`` is implied by ``num_categories``."""

    num_categories: int

    def __post_init__(self) -> None:
        if self.num_categories <= 0:
            raise ValueError(f"onehot realm needs positive num_categories, found {self.num_categories}.")

    @property
    def output_dim(self) -> int:
        return self.num_categories

    def build(self, context: RealmContext) -> torch.nn.Module:
        return OneHot(self.num_categories)


@REALM_ENCODER_REGISTRY.register("sc_flow.feature_mlp")
@dataclass(frozen=True)
class FeatureMLPRealmConfig:
    """Learned projection (MLP) of a looked-up feature vector (fingerprint, precomputed embedding, ...)."""

    input_dim: int
    output_dim: int
    mlp_kwargs: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.output_dim <= 0:
            raise ValueError(
                f"feature_mlp realm needs positive input_dim/output_dim, "
                f"found input_dim={self.input_dim}, output_dim={self.output_dim}."
            )

    def build(self, context: RealmContext) -> torch.nn.Module:
        return init_module_from_dict(dict(self.mlp_kwargs), input_dim=self.input_dim, output_dim=self.output_dim)


def validate_realm_encoder_spec(spec: RealmEncoderSpec | Mapping[str, Any]) -> RealmEncoderSpec:
    """Validate an explicit realm-encoder spec without filling fields or resolving aliases."""
    return REALM_ENCODER_REGISTRY.validate(spec)


def build_realm_encoder(spec: RealmEncoderSpec) -> torch.nn.Module:
    """Build a realm encoder ``nn.Module`` from its spec (everything it needs is in the spec)."""
    return REALM_ENCODER_REGISTRY.build(spec, RealmContext())


def realm_output_dim(spec: RealmEncoderSpec | Mapping[str, Any]) -> int:
    """The realm's post-encoder width (for the SetEncoder projection / decoder sizing) without building."""
    return REALM_ENCODER_REGISTRY.parse(spec).output_dim
