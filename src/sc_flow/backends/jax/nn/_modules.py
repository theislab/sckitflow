import abc
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import flax.linen as nn

__all__ = [
    "BaseModule",
    "MLP",
    "Resnet1d",
]


class BaseModule(abc.ABC, nn.Module):
    """Base class for Neural Networks."""

    @abc.abstractmethod
    def setup(self) -> None:
        """Initialize the module."""
        pass

    @abc.abstractmethod
    def __call__(
        self,
        x: jnp.ndarray,
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        """Performs a forward computation pass on the module.

        :param x: The input tensor to the Neural Network.
        :type x: class: `torch.Tensor`

        :return: The output of the forward computation pass.
        :rtype: class: `torch.nn.Module`
        """

class MLP(BaseModule):
    """
    Class for Multi-Layered Perceptrons with optional batch normalization, layer normalizations and dropout.
    
    Parameters
    ----------
        output_dim
            The output dimensionality for the MLP.
        :param hidden_dims
            (Optional) Sequence of integers indicating the number of neurons for each hidden layer.
            When not provided, it will be initialized to an empty sequence, defaults to `None`.
        activation_cls
            (Optional) Activation function used for the hidden layers of the MLP.
            When not provided, it will be initialized to :class: `flax.linen.activation.relu`, defaults to `None`.
        final_activation_cls
            (Optional) Activation function used for the output layer of the MLP.
            When not provided, it will be initialized as identity, defaults to :obj:`None`.
        use_batchnorm
            (Optional) Whether to use batch normalization after the affine/linear transformation, defaults to `False`.
        batchnorm_epsilon
            (Optional) The :math: `\epsilon` parameter for the denominator of the normalization for numerical stability.
            Only used when :param: `use_batchnorm` is set to :obj:`True`, ignored otherwise. Sets the :attr: `epsilon` attribute
            of :class:`flax.linen.BatchNorm`. Defaults to :math: `10^{-3}`.
        batchnorm_momentum
            (Optional) The value used for the computation of the running statistics for batch normalization.
            Only used when :param: `use_batchnorm` is set to `True`, ignored otherwise. Sets the :attr: `momentum` attribute
            of :class: `flax.linen.BatchNorm`. Defaults to :math: `10^{-2}`.
        batchnorm_use_running_stats
            Whether to use running statistics or simply compute them on each given batch.
            Only used when :param: `use_batchnorm` is set to `True`, ignored otherwise. Sets the :attr: `use_running_stats` attribute
            of :class: `flax.linen.BatchNorm`. Defaults to `True`.
        use_layernorm
            (Optional) Whether to use layer normalization after the affine/linear transformation and optional batch normalization, defaults to `False`.
        layernorm_epsilon
            (Optional) The :math: `\epsilon` parameter for the denominator of the normalization for numerical stability.
            Only used when :param: `use_layernorm` is set to `True`, ignored otherwise. Sets the :attr: `epsilon` attribute
            of :class: `flax.linen.LayerNorm`. Defaults to :math: `10^{-5}`.
        layernorm_bias
            (Optional) Whether to learn a bias term for the element-wise affine transformations. Only used when :param: `use_layernorm`
            and :param: `layernorm_elementwise_affine` are both set to `True`, ignored otherwise. Sets the :attr: `bias` of :class: `flax.linen.LayerNorm`.
            Defaults to `False`.
        dropout_p
            (Optional) The probability for the dropout layer. When positive, a dropout module will be places after each non-linearity.
            Defaults to :math: `0.0`, with no dropout.
        dropout_inplace
            (Optional) Whether to perform dropout in place. Only used when :param: `dropout_p` is positive, ignored otherwise.
            Defaults to `False`.
        activation_cls_kwargs
            (Optional) Key-word arguments for the activation function of the hidden layer of the MLP. When not provided, it will be initialized to
            an empty dictionary. Defaults to `None`.
        final_activation_cls_kwargs
            (Optional) Key-word arguments for the activation function of the output layer of the MLP. When not provided, it will be initialized to
            an empty dictionary. Defaults to `None`.
        bias
        (Optional) Whether to learn a bias term in the linear transformation. Defaults to `True`.
    """

    # input_dim: int
    output_dim: int
    hidden_dims: Sequence[int] | None = None
    activation_cls: Callable | None = None
    final_activation_cls: Callable = None
    use_batchnorm: bool = False
    batchnorm_epsilon: float = 1e-3
    batchnorm_momentum: float = 1e-2
    # batchnorm_affine: bool = True
    batchnorm_use_running_stats: bool = True # SAME AS DROPOUT
    use_layernorm: bool = False
    layernorm_epsilon: float = 1e-5
    # layernorm_elementwise_affine: bool = False
    layernorm_bias: bool = False
    dropout_p: float = 0.0
    dropout_inplace: bool = False
    activation_cls_kwargs: dict[str, Any] | None = None
    final_activation_cls_kwargs: dict[str, Any] | None = None
    bias: bool = True

    def setup(self):
        self._hidden_dims = () if self.hidden_dims is None else self.hidden_dims
        self._activation_cls = nn.relu if self.activation_cls is None else self.activation_cls
        # self._final_activation_cls = torch.nn.Identity if final_activation_cls is None else final_activation_cls
        # self._use_batchnorm = use_batchnorm
        # self._use_layernorm = use_layernorm
        # self._batchnorm_eps = batchnorm_eps
        # self._batchnorm_momentum = batchnorm_momentum
        # self._batchnorm_affine = batchnorm_affine
        # self._batchnorm_track_running_stats = batchnorm_track_running_stats
        # self._layernorm_eps = layernorm_eps
        # self._layernorm_elementwise_affine = layernorm_elementwise_affine
        # self._layernorm_bias = layernorm_bias
        # self._dropout_p = dropout_p
        # self._dropout_inplace = dropout_inplace
        self._activation_cls_kwargs = {} if self.activation_cls_kwargs is None else self.activation_cls_kwargs
        self._final_activation_cls_kwargs = {} if self.final_activation_cls_kwargs is None else self.final_activation_cls_kwargs
        # self._bias = bias

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        for hidden_dim in self._hidden_dims:
            x = nn.Dense(features=hidden_dim, use_bias=self.bias)(x)
            if self.use_batchnorm:
                x = nn.BatchNorm(
                    use_running_average=not train,
                    momentum=self.batchnorm_momentum,
                    epsilon=self.batchnorm_epsilon,
                    use_bias=True,
                    use_scale=True,
                )(x)
            if self.use_layernorm:
                x = nn.LayerNorm(
                    epsilon=self.layernorm_epsilon,
                    use_bias=self.layernorm_bias,
                    use_scale=True,
                )(x)
            x = self._activation_cls(x, **self._activation_cls_kwargs)
            if self.dropout_p > 0.0:
                x = nn.Dropout(rate=self.dropout_p, deterministic=not train)(x)
        x = nn.Dense(features=self.output_dim, use_bias=self.bias)(x)
        if self.final_activation_cls is not None:
            x = self.final_activation_cls(x, **self._final_activation_cls_kwargs)
        return x
    
class Resnet1d(BaseModule):
    """Class for residual connections on 1-dimensional inputs.

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
    
    Parameters
    ----------
        input_dim
            The input dimensionality for the Residual Network.
        output_dim
            The dimensionality of the image space where to project the inputs.
            When not provided, it will be automatically set to :param: `input_dim`. Defaults to `None`.
        num_resnet_layers
            The number of Residual Layers to be stacked in the module.
        hidden_dims
            (Optional) Sequence of integers indicating the number of neurons for each hidden layer.
            When not provided, it will be initialized to an empty sequence, defaults to `None`.
        activation_cls
            (Optional) Activation function used for the Residual Blocks.
            When not provided, it will be initialized to :class: `flax.linen.activation.relu`, defaults to `None`.
        embedding_dim
            The dimensionality of the conditional embedding vector. This embedding is projected and added to the hidden
            state in each residual layer. Defaults to ``32``.
        use_batchnorm
            (Optional) Whether to use batch normalization after the affine/linear transformation, defaults to `False`.
        batchnorm_epsilon
            (Optional) The :math: `\epsilon` parameter for the denominator of the normalization for numerical stability.
            Only used when :param: `use_batchnorm` is set to :obj:`True`, ignored otherwise. Sets the :attr: `epsilon` attribute
            of :class:`flax.linen.BatchNorm`. Defaults to :math: `10^{-3}`.
        batchnorm_momentum
            (Optional) The value used for the computation of the running statistics for batch normalization.
            Only used when :param: `use_batchnorm` is set to `True`, ignored otherwise. Sets the :attr: `momentum` attribute
            of :class: `flax.linen.BatchNorm`. Defaults to :math: `10^{-2}`.
        batchnorm_use_running_stats
            Whether to use running statistics or simply compute them on each given batch.
            Only used when :param: `use_batchnorm` is set to `True`, ignored otherwise. Sets the :attr: `use_running_stats` attribute
            of :class: `flax.linen.BatchNorm`. Defaults to `True`.
        use_layernorm
            (Optional) Whether to use layer normalization after the affine/linear transformation and optional batch normalization, defaults to `False`.
        layernorm_epsilon
            (Optional) The :math: `\epsilon` parameter for the denominator of the normalization for numerical stability.
            Only used when :param: `use_layernorm` is set to `True`, ignored otherwise. Sets the :attr: `epsilon` attribute
            of :class: `flax.linen.LayerNorm`. Defaults to :math: `10^{-5}`.
        layernorm_bias
            (Optional) Whether to learn a bias term for the element-wise affine transformations. Only used when :param: `use_layernorm`
            and :param: `layernorm_elementwise_affine` are both set to `True`, ignored otherwise. Sets the :attr: `bias` of :class: `flax.linen.LayerNorm`.
            Defaults to `False`.
        dropout_p
            (Optional) The probability for the dropout layer. When positive, a dropout module will be places after each non-linearity.
            Defaults to :math: `0.0`, with no dropout.
        dropout_inplace
            (Optional) Whether to perform dropout in place. Only used when :param: `dropout_p` is positive, ignored otherwise.
            Defaults to `False`.
        activation_cls_kwargs
            (Optional) Key-word arguments for the activation function of the hidden layer of the MLP. When not provided, it will be initialized to
            an empty dictionary. Defaults to `None`.
        bias
        (Optional) Whether to learn a bias term in the linear transformation. Defaults to `True`.
    """

    input_dim: int
    output_dim: int
    num_resnet_layers: int
    activation_cls: Callable | None = None
    embedding_dim: int = 32
    use_batchnorm: bool = False
    batchnorm_epsilon: float = 1e-3
    batchnorm_momentum: float = 1e-2
    # batchnorm_affine: bool = True
    batchnorm_use_running_stats: bool = True # SAME AS DROPOUT
    use_layernorm: bool = False
    layernorm_epsilon: float = 1e-5
    # layernorm_elementwise_affine: bool = False
    layernorm_bias: bool = False
    dropout_p: float = 0.0
    dropout_inplace: bool = False
    activation_cls_kwargs: dict[str, Any] | None = None
    final_activation_cls_kwargs: dict[str, Any] | None = None
    bias: bool = True

    def setup(self):
        # self.output_dim = self.input_dim if self.output_dim is None else self.output_dim
        self._activation_cls = nn.relu if self.activation_cls is None else self.activation_cls
        self._activation_cls_kwargs = {} if self.activation_cls_kwargs is None else self.activation_cls_kwargs
        self._final_activation_cls_kwargs = {} if self.final_activation_cls_kwargs is None else self.final_activation_cls_kwargs

    def _apply_block(
        self,
        x: jnp.ndarray,
        train: bool = False
    ) -> jnp.ndarray:
        if self.use_batchnorm:
            x = nn.BatchNorm(
                use_running_average=not train,
                momentum=self.batchnorm_momentum,
                epsilon=self.batchnorm_epsilon,
                use_bias=True,
                use_scale=True,
            )(x)
        if self.use_layernorm:
            x = nn.LayerNorm(
                epsilon=self.layernorm_epsilon,
                use_bias=self.layernorm_bias,
                use_scale=True,
            )(x)
        x = self._activation_cls(x, **self._activation_cls_kwargs)
        if self.dropout_p > 0.0:
            x = nn.Dropout(rate=self.dropout_p, deterministic=not train)(x)
        x = nn.Dense(features=self.output_dim, use_bias=self.bias)(x)

        return x

    @nn.compact
    def __call__(
            self, 
            x: jnp.ndarray, 
            cond: jnp.ndarray, 
            train: bool = False
    ) -> jnp.ndarray:
        """Performs a forward computation pass on the Residual Network.

        parameters
        ----------
        x
            The state which to exert the conditioning on.
        cond
            The conditioning vector.
        train
            Whether the model is in training mode or not. This affects the behaviour of batch normalization
            and dropout layers, when used.
        """         

        for _ in range(self.num_resnet_layers):
            h = self._apply_block(x, train)
            cond_h = nn.Dense(self.output_dim, use_bias=self.bias)(self._activation_cls(cond))
            h = h + cond_h
            h = self._apply_block(h, train)

            x_proj = nn.Dense(self.output_dim)(x) if self.input_dim != self.output_dim else x

            if h.shape != x_proj.shape:
                msg = f"Shape mismatch between hidden condition and state projection. Found {h.shape=} and {x_proj.shape=}"
                raise RuntimeError(msg)
            x = x_proj + h

        return x
    