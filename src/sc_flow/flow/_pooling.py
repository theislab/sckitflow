"""Torch set-pooling components and their portable discriminated specifications.

The attention algorithms are ports of the Flax implementations in
``theislab/cellflow`` (``cellflow.networks._utils``).  They live here rather
than importing cellflow so sc-flow-tools keeps torch-native weights and does
not acquire a cellflow/JAX runtime dependency.
"""

from __future__ import annotations

import abc
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, TypedDict

import torch
from torch.nn import functional as F

__all__ = [
    "BasePooling",
    "MeanPooling",
    "PoolingSpec",
    "SeedAttentionPooling",
    "SumPooling",
    "TokenAttentionPooling",
    "build_pooling",
    "validate_pooling_spec",
]


@dataclass(frozen=True)
class MeanPoolingConfig:
    """Configuration for masked mean pooling."""


@dataclass(frozen=True)
class SumPoolingConfig:
    """Configuration for masked sum pooling."""


def _validate_attention_config(
    *,
    num_heads: int,
    attention_dim: int,
    dropout: float,
    transformer_block: bool,
    layer_norm: bool,
    ff_dim: int | None,
) -> None:
    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, found {num_heads}.")
    if attention_dim <= 0:
        raise ValueError(f"The attention dimension must be positive, found {attention_dim}.")
    if attention_dim % num_heads:
        raise ValueError(f"The attention dimension ({attention_dim}) must be divisible by num_heads ({num_heads}).")
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout must be in [0, 1), found {dropout}.")
    if ff_dim is not None and ff_dim <= 0:
        raise ValueError(f"ff_dim must be positive or None, found {ff_dim}.")
    if not transformer_block and (layer_norm or ff_dim is not None):
        raise ValueError("layer_norm and ff_dim only apply when transformer_block=True.")


@dataclass(frozen=True)
class TokenAttentionPoolingConfig:
    """Configuration for learned-token self-attention pooling."""

    num_heads: int
    qkv_dim: int
    dropout: float
    num_layers: int
    transformer_block: bool
    layer_norm: bool
    ff_dim: int | None

    def __post_init__(self) -> None:
        _validate_attention_config(
            num_heads=self.num_heads,
            attention_dim=self.qkv_dim,
            dropout=self.dropout,
            transformer_block=self.transformer_block,
            layer_norm=self.layer_norm,
            ff_dim=self.ff_dim,
        )
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, found {self.num_layers}.")


@dataclass(frozen=True)
class SeedAttentionPoolingConfig:
    """Configuration for learned-seed cross-attention pooling."""

    num_heads: int
    v_dim: int
    seed_dim: int
    dropout: float
    transformer_block: bool
    layer_norm: bool
    ff_dim: int | None

    def __post_init__(self) -> None:
        _validate_attention_config(
            num_heads=self.num_heads,
            attention_dim=self.v_dim,
            dropout=self.dropout,
            transformer_block=self.transformer_block,
            layer_norm=self.layer_norm,
            ff_dim=self.ff_dim,
        )
        if self.seed_dim <= 0:
            raise ValueError(f"seed_dim must be positive, found {self.seed_dim}.")


PoolingConfig = MeanPoolingConfig | SumPoolingConfig | TokenAttentionPoolingConfig | SeedAttentionPoolingConfig


type JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


class PoolingSpec(TypedDict):
    """The JSON wire representation of a pooling component.

    ``type`` is the stable namespaced discriminator, ``version`` versions that
    component's config schema, and ``config`` contains JSON values only. Every
    field is required, and attention configs must include every version-1
    field. Runtime validation belongs to :func:`validate_pooling_spec`.
    """

    type: str
    version: int
    config: dict[str, JsonValue]


class BasePooling(abc.ABC, torch.nn.Module):
    """Runtime interface for pooling a masked set ``(batch, set, features)``.

    Custom research implementations may subclass this class and be injected
    into :class:`~sc_flow.flow._set_encoder.SetEncoder`.  Doing so grants
    runtime polymorphism only; portable serialization requires a registered
    :class:`PoolingSpec`.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError(f"Pooling dimensions must be positive, found input={input_dim}, output={output_dim}.")
        self._input_dim = input_dim
        self._output_dim = output_dim

    @property
    def input_dim(self) -> int:
        """Required trailing input dimension."""
        return self._input_dim

    @property
    def output_dim(self) -> int:
        """Trailing dimension produced by the pooling operation."""
        return self._output_dim

    def _valid_mask(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor | None:
        if x.ndim != 3:
            raise ValueError(f"Pooling expects x with shape (batch, set, features), found {tuple(x.shape)}.")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Pooling expects input_dim={self.input_dim}, found {x.shape[-1]}.")
        if mask is None:
            return None
        if mask.shape != x.shape[:2]:
            raise ValueError(f"mask must have shape {tuple(x.shape[:2])}, found {tuple(mask.shape)}.")
        if mask.dtype is not torch.bool:
            raise TypeError(f"mask must have dtype torch.bool, found {mask.dtype}.")
        if mask.device != x.device:
            raise ValueError(f"mask and x must be on the same device, found {mask.device} and {x.device}.")
        all_masked = ~mask.any(dim=-1)
        if all_masked.any():
            rows = all_masked.nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(f"Pooling is undefined for all-masked examples; offending batch rows: {rows}.")
        return mask

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Pool valid set elements into one vector per batch example."""


class MeanPooling(BasePooling):
    """Masked arithmetic mean pooling."""

    def __init__(self, input_dim: int) -> None:
        super().__init__(input_dim=input_dim, output_dim=input_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        mask = self._valid_mask(x, mask)
        if mask is None:
            return x.mean(dim=-2)
        weights = mask.unsqueeze(-1).to(dtype=x.dtype)
        return (x * weights).sum(dim=-2) / weights.sum(dim=-2)


class SumPooling(BasePooling):
    """Masked sum pooling."""

    def __init__(self, input_dim: int) -> None:
        super().__init__(input_dim=input_dim, output_dim=input_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        mask = self._valid_mask(x, mask)
        if mask is None:
            return x.sum(dim=-2)
        return (x * mask.unsqueeze(-1).to(dtype=x.dtype)).sum(dim=-2)


def _multihead_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    num_heads: int,
    valid_keys: torch.Tensor | None,
    dropout: float,
    training: bool,
) -> torch.Tensor:
    """Scaled dot-product attention with already-projected Q/K/V tensors."""
    batch, query_len, attention_dim = query.shape
    head_dim = attention_dim // num_heads

    def split_heads(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(batch, -1, num_heads, head_dim).transpose(1, 2)

    q = split_heads(query)
    k = split_heads(key)
    v = split_heads(value)
    attended = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None if valid_keys is None else valid_keys[:, None, None, :],
        dropout_p=dropout if training else 0.0,
    )
    return attended.transpose(1, 2).reshape(batch, query_len, attention_dim)


class _SelfAttentionLayer(torch.nn.Module):
    def __init__(
        self,
        model_dim: int,
        *,
        num_heads: int,
        qkv_dim: int,
        dropout: float,
        transformer_block: bool,
        layer_norm: bool,
        ff_dim: int | None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.transformer_block = transformer_block
        self.q_proj = torch.nn.Linear(model_dim, qkv_dim)
        self.k_proj = torch.nn.Linear(model_dim, qkv_dim)
        self.v_proj = torch.nn.Linear(model_dim, qkv_dim)
        self.out_proj = torch.nn.Linear(qkv_dim, model_dim)
        if transformer_block:
            resolved_ff_dim = 4 * model_dim if ff_dim is None else ff_dim
            self.ff = torch.nn.Sequential(
                torch.nn.Linear(model_dim, resolved_ff_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(resolved_ff_dim, model_dim),
            )
            self.attention_norm = torch.nn.LayerNorm(model_dim) if layer_norm else torch.nn.Identity()
            self.ff_norm = torch.nn.LayerNorm(model_dim) if layer_norm else torch.nn.Identity()

    def forward(self, x: torch.Tensor, valid_tokens: torch.Tensor | None) -> torch.Tensor:
        attended = _multihead_attention(
            self.q_proj(x),
            self.k_proj(x),
            self.v_proj(x),
            num_heads=self.num_heads,
            valid_keys=valid_tokens,
            dropout=self.dropout,
            training=self.training,
        )
        attended = self.out_proj(attended)
        if not self.transformer_block:
            return attended
        x = self.attention_norm(x + F.dropout(attended, p=self.dropout, training=self.training))
        return self.ff_norm(x + F.dropout(self.ff(x), p=self.dropout, training=self.training))


class TokenAttentionPooling(BasePooling):
    """Permutation-invariant self-attention pooling through a learned CLS token."""

    def __init__(self, input_dim: int, config: TokenAttentionPoolingConfig) -> None:
        super().__init__(input_dim=input_dim, output_dim=input_dim)
        self.config = config
        self.class_token = torch.nn.Parameter(torch.empty(1, 1, input_dim))
        torch.nn.init.xavier_uniform_(self.class_token)
        self.layers = torch.nn.ModuleList(
            _SelfAttentionLayer(
                input_dim,
                num_heads=config.num_heads,
                qkv_dim=config.qkv_dim,
                dropout=config.dropout,
                transformer_block=config.transformer_block,
                layer_norm=config.layer_norm,
                ff_dim=config.ff_dim,
            )
            for _ in range(config.num_layers)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        mask = self._valid_mask(x, mask)
        class_token = self.class_token.expand(x.shape[0], -1, -1)
        z = torch.cat((class_token, x), dim=-2)
        if mask is None:
            valid_tokens = None
        else:
            class_mask = torch.ones((x.shape[0], 1), dtype=torch.bool, device=x.device)
            valid_tokens = torch.cat((class_mask, mask), dim=-1)
        for layer in self.layers:
            z = layer(z, valid_tokens)
        return z[:, 0, :]


class SeedAttentionPooling(BasePooling):
    """Permutation-invariant cross-attention pooling through a learned seed."""

    def __init__(self, input_dim: int, config: SeedAttentionPoolingConfig) -> None:
        super().__init__(input_dim=input_dim, output_dim=config.v_dim)
        self.config = config
        self.num_heads = config.num_heads
        self.dropout = config.dropout
        self.transformer_block = config.transformer_block
        self.seed = torch.nn.Parameter(torch.empty(1, 1, config.seed_dim))
        torch.nn.init.xavier_uniform_(self.seed)
        self.q_proj = torch.nn.Linear(config.seed_dim, config.v_dim)
        self.k_proj = torch.nn.Linear(input_dim, config.v_dim)
        self.v_proj = torch.nn.Linear(input_dim, config.v_dim)
        if config.transformer_block:
            resolved_ff_dim = 4 * config.v_dim if config.ff_dim is None else config.ff_dim
            self.ff = torch.nn.Sequential(
                torch.nn.Linear(config.v_dim, resolved_ff_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(resolved_ff_dim, config.v_dim),
            )
            self.attention_norm = torch.nn.LayerNorm(config.v_dim) if config.layer_norm else torch.nn.Identity()
            self.ff_norm = torch.nn.LayerNorm(config.v_dim) if config.layer_norm else torch.nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        mask = self._valid_mask(x, mask)
        query = self.q_proj(self.seed.expand(x.shape[0], -1, -1))
        attended = _multihead_attention(
            query,
            self.k_proj(x),
            self.v_proj(x),
            num_heads=self.num_heads,
            valid_keys=mask,
            dropout=self.dropout,
            training=self.training,
        )
        if self.transformer_block:
            attended = self.attention_norm(query + F.dropout(attended, p=self.dropout, training=self.training))
            attended = self.ff_norm(attended + F.dropout(self.ff(attended), p=self.dropout, training=self.training))
        return attended.squeeze(1)


@dataclass(frozen=True)
class _PoolingRegistration:
    config_type: type[PoolingConfig]
    factory: Callable[[int, PoolingConfig], BasePooling]


def _mean_factory(input_dim: int, config: PoolingConfig) -> BasePooling:
    return MeanPooling(input_dim)


def _sum_factory(input_dim: int, config: PoolingConfig) -> BasePooling:
    return SumPooling(input_dim)


def _token_factory(input_dim: int, config: PoolingConfig) -> BasePooling:
    if not isinstance(config, TokenAttentionPoolingConfig):
        raise TypeError("Internal pooling registry mismatch for token attention.")
    return TokenAttentionPooling(input_dim, config)


def _seed_factory(input_dim: int, config: PoolingConfig) -> BasePooling:
    if not isinstance(config, SeedAttentionPoolingConfig):
        raise TypeError("Internal pooling registry mismatch for seed attention.")
    return SeedAttentionPooling(input_dim, config)


_POOLING_REGISTRY: dict[str, _PoolingRegistration] = {
    "sc_flow.mean": _PoolingRegistration(MeanPoolingConfig, _mean_factory),
    "sc_flow.sum": _PoolingRegistration(SumPoolingConfig, _sum_factory),
    "sc_flow.attention_token": _PoolingRegistration(TokenAttentionPoolingConfig, _token_factory),
    "sc_flow.attention_seed": _PoolingRegistration(SeedAttentionPoolingConfig, _seed_factory),
}


def _parse_pooling_config(spec: PoolingSpec) -> PoolingConfig:
    try:
        registration = _POOLING_REGISTRY[spec["type"]]
    except KeyError:
        raise ValueError(
            f"Unknown pooling type {spec['type']!r}; built-in types are {sorted(_POOLING_REGISTRY)}."
        ) from None
    if spec["version"] != 1:
        raise ValueError(f"Unsupported {spec['type']!r} config version {spec['version']}; supported versions: [1].")
    try:
        return registration.config_type(**spec["config"])
    except TypeError as e:
        raise ValueError(f"Invalid config for pooling type {spec['type']!r}: {e}") from e


def validate_pooling_spec(spec: PoolingSpec | Mapping[str, Any]) -> PoolingSpec:
    """Validate an explicit spec without filling fields or resolving aliases."""
    if not isinstance(spec, Mapping):
        raise TypeError(f"Expected PoolingSpec or mapping; found {type(spec).__name__}.")
    required = {"type", "version", "config"}
    unknown = set(spec) - required
    if unknown:
        raise ValueError(f"Unknown PoolingSpec field(s): {sorted(unknown)}.")
    missing = required - set(spec)
    if missing:
        raise ValueError(f"Missing PoolingSpec field(s): {sorted(missing)}.")
    type_id = spec["type"]
    version = spec["version"]
    raw_config = spec["config"]
    if not isinstance(type_id, str) or not type_id:
        raise TypeError("PoolingSpec['type'] must be a non-empty string.")
    if not isinstance(version, int) or isinstance(version, bool) or version <= 0:
        raise TypeError("PoolingSpec['version'] must be a positive integer.")
    if not isinstance(raw_config, Mapping):
        raise TypeError("PoolingSpec['config'] must be a mapping of JSON values.")
    config_dict = dict(raw_config)
    try:
        json.dumps(config_dict)
    except (TypeError, ValueError) as e:
        raise TypeError("PoolingSpec['config'] must contain JSON-serializable values only.") from e
    copied = PoolingSpec(type=type_id, version=version, config=config_dict)
    config = _parse_pooling_config(copied)
    return PoolingSpec(type=type_id, version=version, config=asdict(config))


def build_pooling(spec: PoolingSpec, *, input_dim: int) -> BasePooling:
    """Build a torch pooling module from a closed built-in spec registry."""
    validated = validate_pooling_spec(spec)
    config = _parse_pooling_config(validated)
    return _POOLING_REGISTRY[validated["type"]].factory(input_dim, config)
