from collections.abc import Callable, Sequence
from typing import Any
from flax.typing import Initializer
from flax.errors import ScopeParamShapeError

import pytest
import jax
import jax.numpy as jnp
import flax.linen as nn

from sc_flow.backends.jax.nn._modules import MLP, Resnet1d


class TestNNModules:
    _input_dim: int = 2
    _output_dim: int = 5
    _batch_size: int = 8
    _embedding_dim: int = 16
    _rng_key: jax.Array = jax.random.PRNGKey(0)

    @pytest.mark.parametrize("hidden_dims", [(), (16,), (16, 16)])
    @pytest.mark.parametrize("activation_cls", [nn.relu])
    @pytest.mark.parametrize("final_activation_cls", [lambda x: x])
    @pytest.mark.parametrize("use_batchnorm", [True, False])
    @pytest.mark.parametrize("batchnorm_epsilon", [1e-3])
    @pytest.mark.parametrize("batchnorm_momentum", [1e-2])
    @pytest.mark.parametrize("batchnorm_bias", [True])
    @pytest.mark.parametrize("batchnorm_scale", [True])
    @pytest.mark.parametrize("batchnorm_bias_init", [nn.initializers.zeros])
    @pytest.mark.parametrize("batchnorm_scale_init", [nn.initializers.ones])
    @pytest.mark.parametrize("use_layernorm", [True, False])
    @pytest.mark.parametrize("layernorm_epsilon", [1e-5])
    @pytest.mark.parametrize("layernorm_bias", [True])
    @pytest.mark.parametrize("layernorm_scale", [True])
    @pytest.mark.parametrize("layernorm_bias_init", [nn.initializers.zeros])
    @pytest.mark.parametrize("layernorm_scale_init", [nn.initializers.ones])
    @pytest.mark.parametrize("dropout_p", [0.0, 0.1])
    @pytest.mark.parametrize("dropout_inplace", [True, False])
    @pytest.mark.parametrize("activation_cls_kwargs", [{}])
    @pytest.mark.parametrize("final_activation_cls_kwargs", [{}])
    @pytest.mark.parametrize("bias", [True, False])
    def test_mlp(
        self,
        hidden_dims: Sequence[int],
        activation_cls: Callable,
        final_activation_cls: Callable,
        use_batchnorm: bool,
        batchnorm_epsilon: float,
        batchnorm_momentum: float,
        batchnorm_bias: bool,
        batchnorm_scale: bool,
        batchnorm_bias_init: Initializer,
        batchnorm_scale_init: Initializer,
        use_layernorm: bool,
        layernorm_epsilon: float,
        layernorm_bias: bool,
        layernorm_scale: bool,
        layernorm_bias_init: Initializer,
        layernorm_scale_init: Initializer,
        dropout_p: float,
        dropout_inplace: bool,
        activation_cls_kwargs: dict[str, Any],
        final_activation_cls_kwargs: dict[str, Any],
        bias: bool,
    ) -> None:
        mlp = MLP(
            self._input_dim,
            self._output_dim,
            hidden_dims=hidden_dims,
            activation_cls=activation_cls,
            final_activation_cls=final_activation_cls,
            use_batchnorm=use_batchnorm,
            batchnorm_epsilon=batchnorm_epsilon,
            batchnorm_momentum=batchnorm_momentum,
            batchnorm_bias=batchnorm_bias,
            batchnorm_scale=batchnorm_scale,
            batchnorm_bias_init=batchnorm_bias_init,
            batchnorm_scale_init=batchnorm_scale_init,
            use_layernorm=use_layernorm,
            layernorm_epsilon=layernorm_epsilon,
            layernorm_bias=layernorm_bias,
            layernorm_scale=layernorm_scale,
            layernorm_bias_init=layernorm_bias_init,
            layernorm_scale_init=layernorm_scale_init,
            dropout_p=dropout_p,
            dropout_inplace=dropout_inplace,
            activation_cls_kwargs=activation_cls_kwargs,
            final_activation_cls_kwargs=final_activation_cls_kwargs,
            bias=bias,
        )

        # case 0: (1, D)
        params = mlp.init(self._rng_key, jnp.zeros((1, self._input_dim)))
        input_tensor = jnp.zeros((1, self._input_dim))
        
        # Test evaluation mode
        output_tensor = mlp.apply(params, input_tensor, train=False)
        assert output_tensor.shape == (1, self._output_dim)
        
        # Test training mode
        if dropout_p > 0.0:
            rng_key = jax.random.PRNGKey(123)
            if use_batchnorm:
                output_tensor, _ = mlp.apply(
                    params, input_tensor, train=True, rngs={'dropout': rng_key}, 
                    mutable=['batch_stats']
                )
            else:
                output_tensor = mlp.apply(params, input_tensor, train=True, rngs={'dropout': rng_key})
        else:
            if use_batchnorm:
                output_tensor, _ = mlp.apply(
                    params, input_tensor, train=True, 
                    mutable=['batch_stats']
                )
            else:
                output_tensor = mlp.apply(params, input_tensor, train=True)
        assert output_tensor.shape == (1, self._output_dim)
        # case 1: (D)
        params = mlp.init(self._rng_key, jnp.zeros((self._input_dim)))
        input_tensor = jnp.zeros(self._input_dim)
        
        output_tensor = mlp.apply(params, input_tensor, train=False)
        assert output_tensor.shape == (self._output_dim,)

        # case 2: (B, D)
        params = mlp.init(self._rng_key, jnp.zeros((self._batch_size, self._input_dim)))
        input_tensor = jnp.zeros((self._batch_size, self._input_dim))
        
        output_tensor = mlp.apply(params, input_tensor, train=False)
        assert output_tensor.shape == (self._batch_size, self._output_dim)
        
        # Test training mode with batch
        if dropout_p > 0.0:
            rng_key = jax.random.PRNGKey(456)
            if use_batchnorm:
                output_tensor, _ = mlp.apply(
                    params, input_tensor, train=True, rngs={'dropout': rng_key}, 
                    mutable=['batch_stats']
                )
            else:
                output_tensor = mlp.apply(params, input_tensor, train=True, rngs={'dropout': rng_key})
        else:
            if use_batchnorm:
                output_tensor, _ = mlp.apply(
                    params, input_tensor, train=True, 
                    mutable=['batch_stats']
                )
            else:
                output_tensor = mlp.apply(params, input_tensor, train=True)
        assert output_tensor.shape == (self._batch_size, self._output_dim)

        # case 3: (1, B, D)
        params = mlp.init(self._rng_key, jnp.zeros((1, self._batch_size, self._input_dim)))
        input_tensor = jnp.zeros((1, self._batch_size, self._input_dim))
        
        output_tensor = mlp.apply(params, input_tensor, train=False)
        assert output_tensor.shape == (1, self._batch_size, self._output_dim)

        # case 4: (B, D + 1) Error
        params = mlp.init(self._rng_key, jnp.zeros((self._batch_size, self._input_dim)))
        input_tensor = jnp.zeros((self._batch_size, self._input_dim + 1))
        
        with pytest.raises(ScopeParamShapeError, match=r".*"):
            output_tensor = mlp.apply(params, input_tensor, train=False)

    @pytest.mark.parametrize("num_resnet_layers", [1, 3])

    @pytest.mark.parametrize("activation_cls", [nn.silu])
    @pytest.mark.parametrize("use_batchnorm", [True, False])
    @pytest.mark.parametrize("batchnorm_epsilon", [1e-3])
    @pytest.mark.parametrize("batchnorm_momentum", [1e-2])
    @pytest.mark.parametrize("batchnorm_bias", [True])
    @pytest.mark.parametrize("batchnorm_scale", [True])
    @pytest.mark.parametrize("batchnorm_bias_init", [nn.initializers.zeros])
    @pytest.mark.parametrize("batchnorm_scale_init", [nn.initializers.ones])
    @pytest.mark.parametrize("use_layernorm", [True, False])
    @pytest.mark.parametrize("layernorm_epsilon", [1e-5])
    @pytest.mark.parametrize("layernorm_bias", [True])
    @pytest.mark.parametrize("layernorm_scale", [True])
    @pytest.mark.parametrize("layernorm_bias_init", [nn.initializers.zeros])
    @pytest.mark.parametrize("layernorm_scale_init", [nn.initializers.ones])
    @pytest.mark.parametrize("dropout_p", [0.0, 0.1])
    @pytest.mark.parametrize("dropout_inplace", [True, False])
    @pytest.mark.parametrize("activation_cls_kwargs", [{}])
    @pytest.mark.parametrize("bias", [True, False])
    def test_resnet1d(
        self,
        num_resnet_layers: int,
        activation_cls: Callable,
        use_batchnorm: bool,
        batchnorm_epsilon: float,
        batchnorm_momentum: float,
        batchnorm_bias: bool,
        batchnorm_scale: bool,
        batchnorm_bias_init: Initializer,
        batchnorm_scale_init: Initializer,
        use_layernorm: bool,
        layernorm_epsilon: float,
        layernorm_bias: bool,
        layernorm_scale: bool,
        layernorm_bias_init: Initializer,
        layernorm_scale_init: Initializer,
        dropout_p: float,
        dropout_inplace: bool,
        activation_cls_kwargs: dict[str, Any],
        bias: bool,
    ):
        resnet = Resnet1d(
            self._input_dim,
            self._output_dim,
            num_resnet_layers,
            activation_cls,
            embedding_dim=self._embedding_dim,
            use_batchnorm=use_batchnorm,
            batchnorm_epsilon=batchnorm_epsilon,
            batchnorm_momentum=batchnorm_momentum,
            batchnorm_bias=batchnorm_bias,
            batchnorm_scale=batchnorm_scale,
            batchnorm_bias_init=batchnorm_bias_init,
            batchnorm_scale_init=batchnorm_scale_init,
            use_layernorm=use_layernorm,
            layernorm_epsilon=layernorm_epsilon,
            layernorm_bias=layernorm_bias,
            layernorm_scale=layernorm_scale,
            layernorm_bias_init=layernorm_bias_init,
            layernorm_scale_init=layernorm_scale_init,
            dropout_p=dropout_p,
            dropout_inplace=dropout_inplace,
            activation_cls_kwargs=activation_cls_kwargs,
            bias=bias,
        )

        # case 0: x.shape = (B, D), cond.shape = (B, K)
        input_tensor = jnp.zeros((self._batch_size, self._input_dim))
        condition = jnp.zeros((self._batch_size, self._embedding_dim))
        params = resnet.init(self._rng_key, input_tensor, condition)

        output_tensor = resnet.apply(params, input_tensor, condition, train=False)
        assert output_tensor.shape == (self._batch_size, self._output_dim)