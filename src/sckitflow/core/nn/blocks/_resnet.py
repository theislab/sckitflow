from collections import OrderedDict
from typing import Any

import torch
from scfit.registry import Component, component

from sckitflow._constants import DEFAULT_NUM_RESNET_LAYERS

__all__ = [
    "ResNet",
    "ResNetConfig",
]


class ResNet(torch.nn.Module):
    r"""Class for residual connections on 1-dimensional inputs.

    The residual network formulation takes the following inputs:
        * Input States: :math: `\boldsymbol{x} \in \mathbb{R}^{n}`.
        * Conditioning Vector: :math: `\boldsymbol{y} \in \mathbb{R}^{m}`.

    We want to map such inputs to conditioned representation :math: `\boldsymbol{x} \in \mathbb{R}^{d}`,
    where we often have :math: `d = n`.

    Each residual layer :math: `k\in\{1, ..., K\}`, with :math: `K` the number of residual layers, consists of:
        * Element-wise non linearity (aka activation function) :math: `f:\R^{D} \rightarrow \R^{D}`.

        * Shallow state projection weights :math: `boldsymbol{W}^{(k)}_{\boldsymbol{X}}\in\mathbb{R}^{n\times d}`, that maps
            state :math: `\boldsymbol{x}^{(k-1)}` to hidden representation :math: `\boldsymbol{h}^{(k)}_{\boldsymbol{X}} = boldsymbol{W}^{(k)}_{\boldsymbol{X}}\cdot f(\boldsymbol{x}^{(k-1)})`.

        * Shallow condition projection weights :math: `boldsymbol{W}^{(k)}_{\boldsymbol{Y}}\in\mathbb{R}^{m\times d}`, that maps
            condition :math: `\boldsymbol{y}` to hidden representation :math: `\boldsymbol{h}^{(k)}_{\boldsymbol{Y}} = boldsymbol{W}^{(k)}_{\boldsymbol{X}}\cdot f(\boldsymbol{y})`.

        * The state and the condition projections :math: `\boldsymbol{h}_{\boldsymbol{X}}^{(k)}` and :math: `\boldsymbol{h}_{\boldsymbol{Y}}^{(k)}` are then summed to obtain
            the hidden conditioned representation :math: `\boldsymbol{h}^{(k)} = \boldsymbol{h}_{\boldsymbol{X}}^{(k)} + \boldsymbol{h}_{\boldsymbol{Y}}^{(k)}`.

        * Shallow hidden projectiion weight :math: `boldsymbol{W}^{(k)}_{\boldsymbol{H}}\in\mathbb{R}^{d\times d}`, that maps hidden conditioned representation
            :math: `\boldsymbol{h}^{(k)}` to latent representation :math: `\boldsymbol{z}^{(k)} = boldsymbol{W}^{(k)}_{\boldsymbol{H}}f(\boldsymbol{h}^{(k)})`.

        * Skip projection weights :math: `boldsymbol{W}^{(k)}_{\text{Skip}}\in\mathbb{R}^{n\times d}` that maps state :math: `\boldsymbol{x}^{(k-1)}`
            to residual skip representation :math: `\boldsymbol{r}^{(k)} = \boldsymbol{W}^{(k)}_{\text{Skip}}\boldsymbol{x}^{(k-1)}`.

    The output of layer :math: `k` is then computed as :math: `\boldsymbol{x}^{(k)} = \boldsymbol{r}^{(k)} + \boldsymbol{z}^{(k)}`.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        *,
        num_resnet_layers: int | None = None,
        output_dim: int | None = None,
        activation_cls: type[torch.nn.Module] | None = None,
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
        self._num_resnet_layers = DEFAULT_NUM_RESNET_LAYERS if num_resnet_layers is None else num_resnet_layers
        self._output_dim = input_dim if output_dim is None else output_dim
        self._activation_cls = torch.nn.SiLU if activation_cls is None else activation_cls
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

        # initializing ResNet
        self._ResNet = torch.nn.Sequential(
            OrderedDict(
                [
                    (f"layer_{layer_id}", self._make_layer(self._input_dim, self._output_dim))
                    if layer_id == 0
                    else (f"layer_{layer_id}", self._make_layer(self._output_dim, self._output_dim))
                    for layer_id in range(self._num_resnet_layers)
                ]
            )
        )

    def _make_block(
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

    def _make_layer(
        self,
        input_dim: int,
        output_dim: int,
    ) -> torch.nn.Module:
        """Initializes a Residual Layer."""
        return torch.nn.ModuleDict(
            [
                ("net1", self._make_block(input_dim, output_dim)),
                (
                    "cond_proj",
                    torch.nn.Sequential(
                        self._activation_cls(**self._activation_cls_kwargs),
                        torch.nn.Linear(self._embedding_dim, output_dim, bias=self._bias),
                    ),
                ),
                (
                    "net2",
                    self._make_block(
                        output_dim,
                        output_dim,
                    ),
                ),
                (
                    "skip_proj",
                    torch.nn.Linear(input_dim, output_dim, bias=self._bias)
                    if input_dim != output_dim
                    else torch.nn.Identity(),
                ),
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Performs a forward computation pass on the Residual Network.

        :param x: The state which to exert the conditioning on.
        :type x: class: `torch.Tensor`

        :param cond: The conditioning vector.
        :type cond: class: `torch.Tensor`
        """
        original_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        cond = cond.reshape(-1, cond.shape[-1])
        for layer in self._ResNet:
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
        """TODO."""
        return self._output_dim


@component("nn.resnet", builds=ResNet)
class ResNetConfig(Component):
    num_resnet_layers: int | None = None
    output_dim: int | None = None
    activation_cls: type[torch.nn.Module] | None = None
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
    bias: bool = True

    def build(
        self,
        input_dim: int,
        embedding_dim: int,
    ) -> ResNet:
        return ResNet(input_dim, embedding_dim, **dict(self))
