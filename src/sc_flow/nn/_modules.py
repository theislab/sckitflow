from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any

import torch

from sc_flow.nn._activation import ActivationId, resolve_activation

__all__ = [
    "FunctionalModule",
    "MLP",
    "Resnet1d",
    "AdaLNZero1d",
]


class FunctionalModule(torch.nn.Module):
    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs)


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] | None = None,
        activation_cls: ActivationId | type[torch.nn.Module] | None = None,
        final_activation_cls: ActivationId | type[torch.nn.Module] | None = None,
        use_batchnorm: bool = False,
        batchnorm_eps: float = 1e-3,
        batchnorm_momentum: float = 1e-2,
        batchnorm_affine: bool = True,
        batchnorm_track_running_stats: bool = True,
        use_layernorm: bool = False,
        layernorm_eps: float = 1e-5,
        layernorm_elementwise_affine: bool = False,
        layernorm_bias: bool = False,
        dropout_p: float = 0.0,
        dropout_inplace: bool = False,
        activation_cls_kwargs: dict[str, Any] | None = None,
        final_activation_cls_kwargs: dict[str, Any] | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._hidden_dims = () if hidden_dims is None else hidden_dims
        # An activation is a serializable enum choice: accept a string id (portable) or a class (runtime-only).
        self._activation_cls = resolve_activation(activation_cls, "relu")
        self._final_activation_cls = resolve_activation(final_activation_cls, "identity")
        self._use_batchnorm = use_batchnorm
        self._use_layernorm = use_layernorm
        self._batchnorm_eps = batchnorm_eps
        self._batchnorm_momentum = batchnorm_momentum
        self._batchnorm_affine = batchnorm_affine
        self._batchnorm_track_running_stats = batchnorm_track_running_stats
        self._layernorm_eps = layernorm_eps
        self._layernorm_elementwise_affine = layernorm_elementwise_affine
        self._layernorm_bias = layernorm_bias
        self._dropout_p = dropout_p
        self._dropout_inplace = dropout_inplace
        self._activation_cls_kwargs = {} if activation_cls_kwargs is None else activation_cls_kwargs
        self._final_activation_cls_kwargs = {} if final_activation_cls_kwargs is None else final_activation_cls_kwargs
        self._bias = bias

        self._mlp = self._build_layers()

    def _build_layer(
        self,
        input_dim: int,
        output_dim: int,
        activation_cls: type[torch.nn.Module],
        activation_cls_kwargs: dict[str, Any],
    ) -> torch.nn.Module:
        layer = [
            ("linear", torch.nn.Linear(input_dim, output_dim, bias=self._bias)),
        ]
        if self._use_batchnorm:
            layer.append(
                (
                    "batchnorm",
                    torch.nn.BatchNorm1d(
                        output_dim,
                        eps=self._batchnorm_eps,
                        momentum=self._batchnorm_momentum,
                        affine=self._batchnorm_affine,
                        track_running_stats=self._batchnorm_track_running_stats,
                    ),
                )
            )
        if self._use_layernorm:
            layer.append(
                (
                    "layernorm",
                    torch.nn.LayerNorm(
                        output_dim,
                        eps=self._layernorm_eps,
                        elementwise_affine=self._layernorm_elementwise_affine,
                        bias=self._layernorm_bias,
                    ),
                )
            )
        layer.append(("activation", activation_cls(**activation_cls_kwargs)))
        if self._dropout_p > 0.0:
            layer.append(
                (
                    "dropout",
                    torch.nn.Dropout(
                        p=self._dropout_p,
                        inplace=self._dropout_inplace,
                    ),
                )
            )
        return torch.nn.Sequential(OrderedDict(layer))

    def _build_layers(
        self,
    ) -> torch.nn.Module:
        layers = []
        input_dim = self._input_dim
        layer_id = 0
        for output_dim in self._hidden_dims:
            layers.append(
                (
                    f"layer_{layer_id}",
                    self._build_layer(
                        input_dim,
                        output_dim,
                        self._activation_cls,
                        self._activation_cls_kwargs,
                    ),
                )
            )
            input_dim = output_dim
            layer_id += 1
        layers.append(
            (
                f"layer_{layer_id}",
                self._build_layer(
                    input_dim,
                    self._output_dim,
                    self._final_activation_cls,
                    self._final_activation_cls_kwargs,
                ),
            )
        )
        return torch.nn.Sequential(OrderedDict(layers))

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        original_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        y = self._mlp(x)
        return y.reshape(*original_shape, -1)


class Resnet1d(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        num_resnet_layers: int,
        output_dim: int | None = None,
        activation_cls: ActivationId | type[torch.nn.Module] | None = None,
        use_batchnorm: bool = False,
        batchnorm_eps: float = 1e-3,
        batchnorm_momentum: float = 1e-2,
        batchnorm_affine: bool = True,
        batchnorm_track_running_stats: bool = True,
        use_layernorm: bool = False,
        layernorm_eps: float = 1e-5,
        layernorm_elementwise_affine: bool = False,
        layernorm_bias: bool = False,
        dropout_p: float = 0.0,
        dropout_inplace: bool = False,
        activation_cls_kwargs: dict[str, Any] | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._embedding_dim = embedding_dim
        self._num_resnet_layers = num_resnet_layers
        self._output_dim = input_dim if output_dim is None else output_dim
        self._activation_cls = resolve_activation(activation_cls, "silu")
        self._use_batchnorm = use_batchnorm
        self._use_layernorm = use_layernorm
        self._batchnorm_eps = batchnorm_eps
        self._batchnorm_momentum = batchnorm_momentum
        self._batchnorm_affine = batchnorm_affine
        self._batchnorm_track_running_stats = batchnorm_track_running_stats
        self._layernorm_eps = layernorm_eps
        self._layernorm_elementwise_affine = layernorm_elementwise_affine
        self._layernorm_bias = layernorm_bias
        self._dropout_p = dropout_p
        self._dropout_inplace = dropout_inplace
        self._activation_cls_kwargs = {} if activation_cls_kwargs is None else activation_cls_kwargs
        self._bias = bias

        self._resnet = self._build_layers()

    def _build_block(
        self,
        input_dim: int,
        output_dim: int,
    ) -> torch.nn.Module:
        layer = []
        if self._use_batchnorm:
            layer.append(
                (
                    "batchnorm",
                    torch.nn.BatchNorm1d(
                        input_dim,
                        eps=self._batchnorm_eps,
                        momentum=self._batchnorm_momentum,
                        affine=self._batchnorm_affine,
                        track_running_stats=self._batchnorm_track_running_stats,
                    ),
                )
            )
        if self._use_layernorm:
            layer.append(
                (
                    "layernorm",
                    torch.nn.LayerNorm(
                        input_dim,
                        eps=self._layernorm_eps,
                        elementwise_affine=self._layernorm_elementwise_affine,
                        bias=self._layernorm_bias,
                    ),
                )
            )
        layer.append(("activation", self._activation_cls(**self._activation_cls_kwargs)))
        if self._dropout_p > 0.0:
            layer.append(
                (
                    "dropout",
                    torch.nn.Dropout(
                        p=self._dropout_p,
                        inplace=self._dropout_inplace,
                    ),
                )
            )
        layer.append(
            (
                "linear",
                torch.nn.Linear(
                    input_dim,
                    output_dim,
                    bias=self._bias,
                ),
            )
        )
        return torch.nn.Sequential(OrderedDict(layer))

    def _build_resnet_layer(
        self,
        input_dim: int,
        output_dim: int,
    ) -> torch.nn.Module:
        return torch.nn.ModuleDict(
            [
                ("net1", self._build_block(input_dim, output_dim)),
                (
                    "cond_proj",
                    torch.nn.Sequential(
                        self._activation_cls(**self._activation_cls_kwargs),
                        torch.nn.Linear(self._embedding_dim, output_dim, bias=self._bias),
                    ),
                ),
                (
                    "net2",
                    self._build_block(
                        output_dim,
                        output_dim,
                    ),
                ),
                (
                    "skip_proj",
                    torch.nn.Linear(input_dim, output_dim) if input_dim != output_dim else torch.nn.Identity(),
                ),
            ]
        )

    def _build_layers(
        self,
    ) -> torch.nn.Module:
        return torch.nn.Sequential(
            OrderedDict(
                [
                    (f"layer_{layer_id}", self._build_resnet_layer(self._input_dim, self._output_dim))
                    if layer_id == 0
                    else (f"layer_{layer_id}", self._build_resnet_layer(self._output_dim, self._output_dim))
                    for layer_id in range(self._num_resnet_layers)
                ]
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        original_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        cond = cond.reshape(-1, cond.shape[-1])
        for layer in self._resnet:
            h = layer["net1"](x)
            h = h + layer["cond_proj"](cond)
            h = layer["net2"](h)

            x_proj = layer["skip_proj"](x)

            if h.shape != x_proj.shape:
                msg = f"Shape mismatch between hidden condition and state projection. Found {h.shape=} and {x_proj.shape=}"
                raise RuntimeError(msg)
            x = x_proj + h
        return x.reshape(*original_shape, -1)

    @property
    def output_dim(
        self,
    ) -> int:
        return self._output_dim


class AdaLNZero1d(torch.nn.Module):
    """DiT-style adaLN-zero trunk: a stack of residual blocks whose LayerNorm is *modulated* by a
    conditioning vector through a **zero-initialized** projection.

    Each block computes ``x = x + gate * MLP(LayerNorm(x) * (1 + scale) + shift)`` where
    ``(scale, shift, gate)`` are produced from ``cond`` by a Linear whose weight **and** bias start at
    zero. So at initialization ``scale = shift = gate = 0`` and every block is the identity
    (``LayerNorm(x) * 1 + 0``, gated by ``0``) — training starts from a stable pass-through and the
    conditioning ramps in as the projection learns. This mirrors CellFlow2's ``adaln_zero`` block.

    The block LayerNorm is non-affine (``elementwise_affine=False``): the learnable affine is supplied by
    the adaLN ``(1 + scale)`` / ``shift`` modulation, as in DiT. Width is constant across blocks (the gated
    residual ``x + gate * h`` requires it), so this is the trunk itself, not a variable-width decoder MLP.
    """

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        num_blocks: int,
        *,
        mlp_ratio: float = 4.0,
        activation_cls: ActivationId | type[torch.nn.Module] | None = None,
        dropout_p: float = 0.0,
        layernorm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError(f"AdaLNZero1d needs at least one block, found num_blocks={num_blocks}.")
        self._dim = int(dim)
        self._num_blocks = int(num_blocks)
        activation_cls = resolve_activation(activation_cls, "silu")
        mlp_hidden = max(1, int(self._dim * mlp_ratio))
        blocks = []
        for block_id in range(self._num_blocks):
            # Zero-init modulation Linear (weight AND bias) => identity block at init (the adaLN-ZERO trick).
            modulation = torch.nn.Linear(cond_dim, 3 * self._dim)
            torch.nn.init.zeros_(modulation.weight)
            torch.nn.init.zeros_(modulation.bias)
            mlp_layers: list[torch.nn.Module] = [
                torch.nn.Linear(self._dim, mlp_hidden),
                activation_cls(),
            ]
            if dropout_p > 0.0:
                mlp_layers.append(torch.nn.Dropout(p=dropout_p))
            mlp_layers.append(torch.nn.Linear(mlp_hidden, self._dim))
            blocks.append(
                torch.nn.ModuleDict(
                    {
                        "norm": torch.nn.LayerNorm(self._dim, eps=layernorm_eps, elementwise_affine=False),
                        "modulation": torch.nn.Sequential(activation_cls(), modulation),
                        "mlp": torch.nn.Sequential(*mlp_layers),
                    }
                )
            )
        self._blocks = torch.nn.ModuleList(blocks)

    @property
    def output_dim(self) -> int:
        return self._dim

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        cond = cond.reshape(-1, cond.shape[-1])
        for block in self._blocks:
            scale, shift, gate = block["modulation"](cond).chunk(3, dim=-1)
            h = block["norm"](x) * (1.0 + scale) + shift
            h = block["mlp"](h)
            x = x + gate * h
        return x.reshape(*original_shape, -1)
