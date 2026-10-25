from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

import torch
from scfit.registry import Component, component

from sckitflow._types import ActivationId

__all__ = [
    "MLP",
    "MLPConfig",
    "MLPEncoderConfig",
]


class MLP(torch.nn.Module):
    """Class for Multi-Layered Perceptrons with optional batch normalization, layer normalizations and dropout."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dims: Sequence[int] | None = None,
        activation_cls: type[torch.nn.Module] | None = None,
        final_activation_cls: type[torch.nn.Module] = None,
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
        self._activation_cls = torch.nn.ReLU if activation_cls is None else activation_cls
        self._final_activation_cls = torch.nn.Identity if final_activation_cls is None else final_activation_cls
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

        # initializing mlp
        layers = []
        input_dim = self._input_dim
        layer_id = 0
        for output_dim in self._hidden_dims:
            layers.append(
                (
                    f"layer_{layer_id}",
                    self._make_layer(
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
                self._make_layer(
                    input_dim,
                    self._output_dim,
                    self._final_activation_cls,
                    self._final_activation_cls_kwargs,
                    final=True,
                ),
            )
        )
        self._mlp = torch.nn.Sequential(OrderedDict(layers))

    def _make_layer(
        self,
        input_dim: int,
        output_dim: int,
        activation_cls: type[torch.nn.Module],
        activation_cls_kwargs: dict[str, Any],
        *,
        final: bool = False,
    ) -> torch.nn.Module:
        layer = [
            ("linear", torch.nn.Linear(input_dim, output_dim, bias=self._bias)),
        ]
        if self._use_batchnorm and not final:
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
        if self._use_layernorm and not final:
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
        if self._dropout_p > 0.0 and not final:
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

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        r"""Forward computation pass on the input tensor :math: `\boldsymbol{x}`.

        :param x: The input tensor to the :class: `MLP`. Should be of shape :math: `(\star, D_{Input})`,
            where :math: `\star` denotes any arbitrary number of leading dimensions and :math: `D_{Input}` denotes the
            input dimensionality set in :attr: `self._input_dim`.
        :type x: class: `torch.Tensor`

        :return: A tensor of shape :math: `(\star, D_{Output})`, where :math: `\star` denotes same arbitrary number of leading dimensions
            as the input and :math: `D_{Output}` denotes the output dimensionality set in :attr: `self._output_dim`.
        :rtype: class: `torch.Tensor`
        """
        original_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        y = self._mlp(x)
        return y.reshape(*original_shape, -1)


_ACTIVATIONS: dict[ActivationId, type[torch.nn.Module]] = {
    "relu": torch.nn.ReLU,
    "leaky_relu": torch.nn.LeakyReLU,
    "elu": torch.nn.ELU,
    "gelu": torch.nn.GELU,
    "silu": torch.nn.SiLU,
    "tanh": torch.nn.Tanh,
    "sigmoid": torch.nn.Sigmoid,
    "softplus": torch.nn.Softplus,
}


@component("nn.mlp", builds=MLP)
class MLPConfig(Component):
    hidden_dims: tuple[int, ...] | None = None
    activation_cls: ActivationId | None = None
    final_activation_cls: ActivationId | None = None
    use_batchnorm: bool = False
    batchnorm_eps: float = 1e-3
    batchnorm_momentum: float = 1e-2
    batchnorm_affine: bool = True
    batchnorm_track_running_stats: bool = True
    use_layernorm: bool = False
    layernorm_eps: float = 1e-5
    layernorm_elementwise_affine: bool = False
    layernorm_bias: bool = False
    dropout_p: float = 0.0
    dropout_inplace: bool = False
    activation_cls_kwargs: dict[str, Any] | None = None
    final_activation_cls_kwargs: dict[str, Any] | None = None
    bias: bool = True

    def mlp_kwargs(self) -> dict[str, Any]:
        """The keyword arguments of :class:`MLP`, with activation names mapped to their classes."""
        kwargs = dict(self)
        for key in ("activation_cls", "final_activation_cls"):
            if kwargs[key] is not None:
                kwargs[key] = _ACTIVATIONS[kwargs[key]]
        if "hidden_dims" in kwargs and kwargs["hidden_dims"] is None:
            kwargs["hidden_dims"] = ()
        return kwargs

    def build(self, input_dim: int, output_dim: int) -> MLP:
        return MLP(input_dim, output_dim, **self.mlp_kwargs())


@component("nn.mlp_encoder")
class MLPEncoderConfig(Component):
    output_dim: int
    mlp: MLPConfig

    def build(self, input_dim: int) -> MLP:
        return self.mlp.build(input_dim, self.output_dim)
