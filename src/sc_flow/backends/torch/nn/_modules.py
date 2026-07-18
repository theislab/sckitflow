import abc
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any

import torch

from sc_flow._constants import DEFAULT_NUM_RESNET_LAYERS

__all__ = [
    "BaseModule",
    "FunctionalModule",
    "MLP",
    "Resnet1d",
]


class BaseModule(abc.ABC, torch.nn.Module):
    """Base class for Neural Networks."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @abc.abstractmethod
    def _make_modules(
        self,
    ) -> torch.nn.Module:
        """Initializes the module."""

    @abc.abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Performs a forward computation pass on the module.

        :param x: The input tensor to the Neural Network.
        :type x: class: `torch.Tensor`

        :return: The output of the forward computation pass.
        :rtype: class: `torch.nn.Module`
        """


class FunctionalModule(BaseModule):
    """Class for wrapping :class: `torch.nn.Modules` around callables."""

    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """TODO."""
        super().__init__()
        self.fn = fn

        self._identity = self._make_modules()

    def _make_modules(self):
        """TODO."""
        return torch.nn.Identity()

    def forward(self, x, *args, **kwargs):
        """TODO."""
        out = self.fn(x, *args, **kwargs)
        return self._identity(out)


class MLP(BaseModule):
    """Class for Multi-Layered Perceptrons with optional batch normalization, layer normalizations and dropout."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
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
        r"""Initializes the Multi-Layered Perceptron.

        :param input_dim: The input dimensionality for the MLP.
        :type input_dim: class: `int`

        :param output_dim: The output dimensionality for the MLP.
        :type output_dim: class: `int`

        :param hidden_dims: (Optional) Sequence of integers indicating the number of neurons for each hidden layer.
            When not provided, it will be initialized to an empty sequence, defaults to `None`.
        :type hidden_dims: `Sequence[int] | None`

        :param activation_cls: (Optional) :class: `torch.nn.Module` used as activation function for the hidden layers of the MLP.
            When not provided, it will be initialized to :class: `torch.nn.ReLU`, defaults to `None`.
        :type activation_cls: class: `type[torch.nn.Module]`

        :param final_activation_cls: (Optional) :class: `torch.nn.Module` used as activation function for the output layer of the MLP.
            When not provided, it will be initialized to :class: `torch.nn.Identity`, defaults to `None`.
        :type final_activation_cls: class: `type[torch.nn.Module]`

        :param use_batchnorm: (Optional) Whether to use batch normalization after the affine/linear transformation, defaults to `False`.
        :type use_batchnorm: class: `bool`

        :param batchnorm_eps: (Optional) The :math: `\epsilon` parameter for the denominator of the normalization for numerical stability.
            Only used when :param: `use_batchnorm` is set to `True`, ignored otherwise. Sets the :attr: `eps` attribute
            of :class: `torch.nn.BatchNorm1d`. Defaults to :math: `10^{-3}`.
        :type batchnorm: class: `float`

        :param batchnorm_momentum: (Optional) The value used for the computation of the running statistics for batch normalization.
            Only used when :param: `use_batchnorm` is set to `True`, ignored otherwise. Sets the :attr: `momentum` attribute
            of :class: `torch.nn.BatchNorm1d`. Defaults to :math: `10^{-2}`.
        :type batchnorm_momentum: class: `float`

        :param batchnorm_affine: (Optional) Whether to use learnable affine parameters for the batch normalization module.
            Only used when :param: `use_batchnorm` is set to `True`, ignored otherwise. Sets the :attr: `affine` attribute
            of :class: `torch.nn.BatchNorm1d`. Defaults to `True`.
        :type batchnorm_affine: class: `bool`

        :param batchnorm_track_running_stats: Whether to track running statistics or simply compute them on each given batch.
            Only used when :param: `use_batchnorm` is set to `True`, ignored otherwise. Sets the :attr: `track_running_stats` attribute
            of :class: `torch.nn.BatchNorm1d`. Defaults to `True`.
        :type batchnorm_track_running_stats: class: `bool`

        :param use_layernorm: (Optional) Whether to use layer normalization after the affine/linear transformation and optional batch normalization, defaults to `False`.
        :type use_layernorm: class: `bool`

        :param layernorm_eps: (Optional) The :math: `\epsilon` parameter for the denominator of the normalization for numerical stability.
            Only used when :param: `use_layernorm` is set to `True`, ignored otherwise. Sets the :attr: `eps` attribute
            of :class: `torch.nn.LayerNorm`. Defaults to :math: `10^{-5}`.
        :type layernorm_eps: class: `float`

        :param layernorm_elementwise_affine: (Optional) Whether to use learnable element-wise affine parameters for the layer normalization module.
            Only used when :param: `use_layernorm` is set to `True`, ignored otherwise. Sets the :attr: `elementwise_affine` attribute
            of :class: `torch.nn.Layernorm`. Defaults to `False`.
        :type layernorm_elementwise_affine: class: `bool`

        :param layernorm_bias: (Optional) Whether to learn a bias term for the element-wise affine transformations. Only used when :param: `use_layernorm`
            and :param: `layernorm_elementwise_affine` are both set to `True`, ignored otherwise. Sets the :attr: `bias` of :class: `torch.nn.LayerNorm`.
            Defaults to `False`.
        :type layernorm_bias: class: `bool`

        :param dropout_p: (Optional) The probability for the dropout layer. When positive, a dropout module will be places after each non-linearity.
            Defaults to :math: `0.0`, with no dropout.
        :type dropout_p: class: `float`

        :param dropout_inplace: (Optional) Whether to perform dropout in place. Only used when :param: `dropout_p` is positive, ignored otherwise.
            Defaults to `False`.
        :type dropout_inplace: class: `bool`

        :param activation_cls_kwargs: (Optional) Key-word arguments for the activation function of the hidden layer of the MLP. When not provided, it will be initialized to
            an empty dictionary. Defaults to `None`.
        :type activation_cls_kwargs: class: `dict[str, Any] | None`

        :param final_activation_cls_kwargs: (Optional) Key-word arguments for the activation function of the output layer of the MLP. When not provided, it will be initialized to
            an empty dictionary. Defaults to `None`.
        :type final_activation_cls_kwargs: class: `dict[str, Any] | None`

        :param bias: (Optional) Whether to learn a bias term in the linear transformation. Defaults to `True`.
        :type bias: class: `bool`
        """
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
        self._mlp = self._make_modules()

    def _make_layer(
        self,
        input_dim: int,
        output_dim: int,
        activation_cls: type[torch.nn.Module],
        activation_cls_kwargs: dict[str, Any],
    ) -> torch.nn.Module:
        """Initializes a single layer.

        :param input_dim: The input dimensionality of the layer.
        :type input_dim: class: `int`

        :param output_dim: The output dimensionality of the layer.
        :type output_dim: class: `int`

        :param activation_cls: Class for the activation function of the layer.
        :type activation_cls: class: `type[torch.nn.Module]`

        :param activation_cls_kwargs: Key-word arguments for the activation of the layer.
        :type activation_cls_kwargs: class: `dict[str, Any]`

        :return: A :class: `torch.nn.Sequential` module for the current layer.
        :rtype: class: `torch.nn.Module`
        """
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

    def _make_modules(
        self,
    ) -> torch.nn.Module:
        """Initializes the MLP.

        :return: A :class: `torch.nn.Sequential` module for the whole MLP.
        :rtype: class: `torch.nn.Module`
        """
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
                ),
            )
        )
        return torch.nn.Sequential(OrderedDict(layers))

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


class Resnet1d(BaseModule):
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
        r"""Initializes the Residual Network.

        :param input_dim: The input dimensionality for the Residual Network.
        :type input_dim: class: `int`

        :param embedding_dim: The dimensionality of the conditional embedding vector. This embedding is projected and added to the hidden
            state in each residual layer.
        :type embedding_dim: class: `int`

        :param num_resnet_layers: The number of Residual Layers to be stacked in the module.
        :type num_resnet_layers: class: `int`

        :param output_dim: (Optional) The dimensionality of the image space where to project the inputs.
            When not provided, it will be automatically set to :param: `input_dim`. Defaults to `None`.
        :type output_dim: class: `int | None`

        :param activation_cls: (Optional) :class: `torch.nn.Module` used as non-linearity for the Residual Block.
            When not provided, it will be initialized to :class: `torch.nn.SiLU`, defaults to `None`.
        :type activation_cls: class: `type[torch.nn.Module]`

        :param use_batchnorm: (Optional) Whether to use batch normalization before the affine/linear transformation in the residual block, defaults to `False`.
        :type use_batchnorm: class: `bool`

        :param batchnorm_eps: (Optional) The :math: `\epsilon` parameter for the denominator of the normalization for numerical stability.
            Only used when :param: `use_batchnorm` is set to `True`, ignored otherwise. Sets the :attr: `eps` attribute
            of :class: `torch.nn.BatchNorm1d`. Defaults to :math: `10^{-3}`.
        :type batchnorm: class: `float`

        :param batchnorm_momentum: (Optional) The value used for the computation of the running statistics for batch normalization.
            Only used when :param: `use_batchnorm` is set to `True`, ignored otherwise. Sets the :attr: `momentum` attribute
            of :class: `torch.nn.BatchNorm1d`. Defaults to :math: `10^{-2}`.
        :type batchnorm_momentum: class: `float`

        :param batchnorm_affine: (Optional) Whether to use learnable affine parameters for the batch normalization module.
            Only used when :param: `use_batchnorm` is set to `True`, ignored otherwise. Sets the :attr: `affine` attribute
            of :class: `torch.nn.BatchNorm1d`. Defaults to `True`.
        :type batchnorm_affine: class: `bool`

        :param batchnorm_track_running_stats: Whether to track running statistics or simply compute them on each given batch.
            Only used when :param: `use_batchnorm` is set to `True`, ignored otherwise. Sets the :attr: `track_running_stats` attribute
            of :class: `torch.nn.BatchNorm1d`. Defaults to `True`.
        :type batchnorm_track_running_stats: class: `bool`

        :param use_layernorm: (Optional) Whether to use layer normalization before the affine/linear transformation and after optional batch normalization in the residual block, defaults to `False`.
        :type use_layernorm: class: `bool`

        :param layernorm_eps: (Optional) The :math: `\epsilon` parameter for the denominator of the normalization for numerical stability.
            Only used when :param: `use_layernorm` is set to `True`, ignored otherwise. Sets the :attr: `eps` attribute
            of :class: `torch.nn.LayerNorm`. Defaults to :math: `10^{-5}`.
        :type layernorm_eps: class: `float`

        :param layernorm_elementwise_affine: (Optional) Whether to use learnable element-wise affine parameters for the layer normalization module.
            Only used when :param: `use_layernorm` is set to `True`, ignored otherwise. Sets the :attr: `elementwise_affine` attribute
            of :class: `torch.nn.Layernorm`. Defaults to `False`.
        :type layernorm_elementwise_affine: class: `bool`

        :param layernorm_bias: (Optional) Whether to learn a bias term for the element-wise affine transformations. Only used when :param: `use_layernorm`
            and :param: `layernorm_elementwise_affine` are both set to `True`, ignored otherwise. Sets the :attr: `bias` of :class: `torch.nn.LayerNorm`.
            Defaults to `False`.
        :type layernorm_bias: class: `bool`

        :param dropout_p: (Optional) The probability for the dropout layer. When positive, a dropout module will be places before the linear/affine transformation.
            Defaults to :math: `0.0`, with no dropout.
        :type dropout_p: class: `float`

        :param dropout_inplace: (Optional) Whether to perform dropout in place. Only used when :param: `dropout_p` is positive, ignored otherwise.
            Defaults to `False`.
        :type dropout_inplace: class: `bool`

        :param activation_cls_kwargs: (Optional) Key-word arguments for the activation function of the hidden layer of the MLP. When not provided, it will be initialized to
            an empty dictionary. Defaults to `None`.
        :type activation_cls_kwargs: class: `dict[str, Any] | None`

        :param bias: (Optional) Whether to learn a bias term in the linear transformation. Defaults to `True`.
        :type bias: class: `bool`
        """
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

        # initializing resnet
        self._resnet = self._make_modules()

    def _make_block(
        self,
        input_dim: int,
        output_dim: int,
    ) -> torch.nn.Module:
        """Initializes a projection block given an input and an ouput dimensionality.

        :param input_dim: The input dimensionality for the current block.
        :type input_dim: class: `int`

        :param output_dim: The output dimensionality for the current block.
        :type output_dim: class: `int`

        :return: An initialized projection block.
        :rtype: class: `torch.nn.Module`
        """
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

    def _make_resnet_layer(
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
                    torch.nn.Linear(input_dim, output_dim) if input_dim != output_dim else torch.nn.Identity(),
                ),
            ]
        )

    def _make_modules(
        self,
    ) -> torch.nn.Module:
        """Initializes the stack of Residual Layers."""
        return torch.nn.Sequential(
            OrderedDict(
                [
                    (f"layer_{layer_id}", self._make_resnet_layer(self._input_dim, self._output_dim))
                    if layer_id == 0
                    else (f"layer_{layer_id}", self._make_resnet_layer(self._output_dim, self._output_dim))
                    for layer_id in range(self._num_resnet_layers)
                ]
            )
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
        """TODO."""
        return self._output_dim
