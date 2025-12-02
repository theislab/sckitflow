import pytest
import jax.numpy as jnp
import jax
from flax import linen as nn
from flax.errors import ScopeParamNotFoundError

from sc_flow._types import LayersDict, NestedLayersDict
from sc_flow.backends.jax.nn._set_encoder import SetEncoder

# dimensions
batch_size = 32
n_combs = 3
output_dim = 16
condition0_input_dim = 4
condition0_output_dim = 2
condition1_input_dim = 8
condition1_output_dim = 4

# dictionaries
input_layers_single_condition = {
    "condition0": {
        "input_dim": condition0_input_dim,
        "output_dim": condition0_output_dim,
    },
}
input_layers_double_condition = {
    "condition0": {
        "input_dim": condition0_input_dim,
        "output_dim": condition0_output_dim,
    },
    "condition1": {
        "input_dim": condition1_input_dim,
        "output_dim": condition1_output_dim,
    },
}


class TestSetEncoder:
    @pytest.mark.parametrize("input_layers", [input_layers_single_condition, input_layers_double_condition])
    @pytest.mark.parametrize("pooling_mode", ["mean", "sum"])
    @pytest.mark.parametrize("output_layers_kwargs", [{}])
    def test_set_encoder_forward(
        self,
        input_layers: NestedLayersDict,
        pooling_mode: str,
        output_layers_kwargs: LayersDict,
    ) -> None:
        # initializing encoder

        encoder = SetEncoder(
            input_layers,
            output_dim,
            pooling_mode=pooling_mode,
            pooling_kwargs=None,
            output_layers_kwargs=output_layers_kwargs,
        )

        if len(input_layers) == 1:
            condition_dict = {
                "condition0": jnp.zeros((batch_size, n_combs, condition0_input_dim)),
            }
        else:
            condition_dict = {
                "condition0": jnp.zeros((batch_size, n_combs, condition0_input_dim)),
                "condition1": jnp.zeros((batch_size, n_combs, condition1_input_dim)),
            }

        rng = jax.random.PRNGKey(0)
        params = encoder.init(rng, condition_dict)

        encoded_condition = encoder.apply(params, condition_dict)
        assert encoded_condition.shape == (batch_size, output_dim)

        # Double condition case.
        if len(input_layers) == 1:
            condition_dict = {
                "condition0": jnp.zeros((batch_size, n_combs, condition0_input_dim)),
                "condition1": jnp.zeros((batch_size, n_combs, condition1_input_dim)),
            }
            with pytest.raises(KeyError):
                encoded_condition = encoder.apply(params, condition_dict)
