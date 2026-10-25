import abc
from collections.abc import Callable
from typing import ClassVar

import torch
from scfit.registry import Component, component

__all__ = [
    "BasePoolingLayer",
    "TokenAttentionPooling",
    "SeedAttentionPooling",
    "AttentionPoolingConfig",
    "TokenAttentionPoolingConfig",
    "SeedAttentionPoolingConfig",
    "PoolingLayerConfig",
    "MeanPoolingConfig",
    "SumPoolingConfig",
]


class BasePoolingLayer(abc.ABC, torch.nn.Module):
    @property
    @abc.abstractmethod
    def output_dim(self) -> int:
        raise NotImplementedError


class _AttentionPooling(torch.nn.Module):
    """Pools a set ``[..., n, input_dim]`` with one learned query attending over it."""

    def __init__(
        self, input_dim: int, *, query_dim: int, qkv_dim: int, num_heads: int, dropout_rate: float, token: bool
    ) -> None:
        super().__init__()
        if qkv_dim % num_heads:
            raise ValueError(f"`qkv_dim` ({qkv_dim}) must be divisible by `num_heads` ({num_heads}).")
        self._num_heads = num_heads
        self._dropout_rate = dropout_rate
        self._token = token
        self._query = torch.nn.Parameter(torch.randn(1, 1, query_dim) * query_dim**-0.5)
        self._q = torch.nn.Linear(query_dim, qkv_dim)
        self._k = torch.nn.Linear(input_dim, qkv_dim)
        self._v = torch.nn.Linear(input_dim, qkv_dim)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.unflatten(-1, (self._num_heads, -1)).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-2]
        x = x.reshape(-1, *x.shape[-2:])
        query = self._query.expand(x.shape[0], -1, -1)
        if self._token:
            x = torch.concatenate([query, x], dim=1)  # the token is part of the set it attends over
        out = torch.nn.functional.scaled_dot_product_attention(
            self._heads(self._q(query)),
            self._heads(self._k(x)),
            self._heads(self._v(x)),
            dropout_p=self._dropout_rate if self.training else 0.0,
        )
        return out.transpose(1, 2).flatten(-2)[:, 0].reshape(*lead, -1)


class TokenAttentionPooling(BasePoolingLayer, _AttentionPooling):
    """Prepends a learned token to the set and keeps what it attends to, as CellFlow's ``attention_token``.

    :param input_dim: Dimensionality of the set elements, and of the output.
    :param num_heads: Number of attention heads.
    :param qkv_dim: Dimensionality of queries, keys and values, split across the heads.
    :param dropout_rate: Dropout on the attention weights while training.
    """

    def __init__(self, input_dim: int, num_heads: int = 8, qkv_dim: int = 64, dropout_rate: float = 0.0) -> None:
        super().__init__(
            input_dim, query_dim=input_dim, qkv_dim=qkv_dim, num_heads=num_heads, dropout_rate=dropout_rate, token=True
        )
        self._out = torch.nn.Linear(qkv_dim, input_dim)
        self._output_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._out(super().forward(x))

    @property
    def output_dim(self) -> int:
        return self._output_dim


class SeedAttentionPooling(BasePoolingLayer, _AttentionPooling):
    """Pooling by multi-head attention with a learned seed (Set Transformer's PMA), as CellFlow's ``attention_seed``.

    :param input_dim: Dimensionality of the set elements.
    :param num_heads: Number of attention heads.
    :param v_dim: Dimensionality of queries, keys and values, split across the heads, and of the output.
    :param seed_dim: Dimensionality of the learned seed.
    :param dropout_rate: Dropout on the attention weights while training.
    """

    # ponytail: no `transformer_block` / `layer_norm` (off by default in CellFlow); add when a model needs them.
    def __init__(
        self, input_dim: int, num_heads: int = 8, v_dim: int = 64, seed_dim: int = 64, dropout_rate: float = 0.0
    ) -> None:
        super().__init__(
            input_dim, query_dim=seed_dim, qkv_dim=v_dim, num_heads=num_heads, dropout_rate=dropout_rate, token=False
        )
        self._output_dim = v_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim


class AttentionPoolingConfig(Component):
    num_heads: int = 8
    dropout_rate: float = 0.0


@component("nn.pooling.token_attention", builds=TokenAttentionPooling)
class TokenAttentionPoolingConfig(AttentionPoolingConfig):
    qkv_dim: int = 64

    def build(self, input_dim: int) -> TokenAttentionPooling:
        return TokenAttentionPooling(input_dim, **dict(self))


@component("nn.pooling.seed_attention", builds=SeedAttentionPooling)
class SeedAttentionPoolingConfig(AttentionPoolingConfig):
    v_dim: int = 64
    seed_dim: int = 64

    def build(self, input_dim: int) -> SeedAttentionPooling:
        return SeedAttentionPooling(input_dim, **dict(self))


class _FunctionalModule(BasePoolingLayer, torch.nn.Module):
    def __init__(self, input_dim: int, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self._fn = fn
        # We don't really need the ``self._input_dim`` attribute
        self._output_dim = input_dim
        # The identity module is required to register the parameters of modules
        # using this functional component; an error is thrown otherwise. Hence,
        # despite this being a no-op, we need to keep it nevertheless.
        self._net = torch.nn.Identity()

    def forward(self, x: torch.Tensor):
        return self._net(self._fn(x))

    @property
    def output_dim(self) -> int:
        return self._output_dim


class _AggregationPoolingConfig(Component):
    fn: ClassVar[Callable[[torch.Tensor, int], torch.Tensor]]

    def build(self, input_dim: int) -> torch.nn.Module:
        def _wrapped_fn(x: torch.Tensor):
            return type(self).fn(x, dim=-2)

        return _FunctionalModule(input_dim, _wrapped_fn)


@component("nn.pooling.mean")
class MeanPoolingConfig(_AggregationPoolingConfig):
    fn: ClassVar[Callable[[torch.Tensor, int], torch.Tensor]] = staticmethod(torch.mean)


@component("nn.pooling.sum")
class SumPoolingConfig(_AggregationPoolingConfig):
    fn: ClassVar[Callable[[torch.Tensor, int], torch.Tensor]] = staticmethod(torch.sum)


PoolingLayerConfig = TokenAttentionPoolingConfig | SeedAttentionPoolingConfig | MeanPoolingConfig | SumPoolingConfig
