import pytest
import jax.numpy as jnp
import jax

from sc_flow.backends.jax._types import ArrayLike
from sc_flow.backends.jax.nn._conditioning_layers import (
    ConcatConditioning,
    Resnet1dConditioning,
    make_custom_conditioning_layer,
    get_conditioning_layer,
)



# defining variables
batch_size = 8
num_samples = 50
num_feats = 16
num_time_feats = 6
num_cond_feats = 8


class TestConditioningLayers:


    @pytest.mark.parametrize("concat_condition", [True, False])
    def test_concat_conditioning(
        self,
        concat_condition: bool
    ) -> None:
        
        if concat_condition:
            conditioning = ConcatConditioning(
                num_feats,
                num_time_feats,
                num_cond_feats,
            )
        else:
            conditioning = ConcatConditioning(
                num_feats,
                num_time_feats,
            )

        # case 0
        t = jnp.zeros((num_time_feats, ))
        cond = jnp.zeros((num_cond_feats, ))
        x = jnp.zeros((batch_size, num_feats))
        if concat_condition:
            out = conditioning(t, x, cond)
            assert out.shape == ((batch_size, num_feats + num_time_feats + num_cond_feats))
            assert out.shape[-1] == conditioning.output_dim
        else:
            out = conditioning(t, x)
            assert out.shape == ((batch_size, num_feats + num_time_feats))
            assert out.shape[-1] == conditioning.output_dim

        # case 1
        t = jnp.zeros((1, num_time_feats))
        cond = jnp.zeros((1, num_cond_feats))
        x = jnp.zeros((batch_size, num_feats))
        if concat_condition:
            out = conditioning(t, x, cond)
            assert out.shape == ((batch_size, num_feats + num_time_feats + num_cond_feats))
            assert out.shape[-1] == conditioning.output_dim
        else:
            out = conditioning(t, x)
            assert out.shape == ((batch_size, num_feats + num_time_feats))
            assert out.shape[-1] == conditioning.output_dim

        # case 2
        t = jnp.zeros((batch_size, num_time_feats))
        cond = jnp.zeros((batch_size, num_cond_feats))
        x = jnp.zeros((batch_size, num_feats))
        if concat_condition:
            out = conditioning(t, x, cond)
            assert out.shape == ((batch_size, num_feats + num_time_feats + num_cond_feats))
            assert out.shape[-1] == conditioning.output_dim
        else:
            out = conditioning(t, x)
            assert out.shape == ((batch_size, num_feats + num_time_feats))
            assert out.shape[-1] == conditioning.output_dim

        # case 3
        t = jnp.zeros((num_time_feats, ))
        cond = jnp.zeros((num_cond_feats, ))
        x = jnp.zeros((batch_size, num_samples, num_feats))
        if concat_condition:
            out = conditioning(t, x, cond)
            assert out.shape == ((batch_size, num_samples, num_feats + num_time_feats + num_cond_feats))
            assert out.shape[-1] == conditioning.output_dim
        else:
            out = conditioning(t, x)
            assert out.shape == ((batch_size, num_samples, num_feats + num_time_feats))
            assert out.shape[-1] == conditioning.output_dim

        # case 4
        t = jnp.zeros((1, num_time_feats))
        cond = jnp.zeros((1, num_cond_feats))
        x = jnp.zeros((batch_size, num_samples, num_feats))
        if concat_condition:
            out = conditioning(t, x, cond)
            assert out.shape == ((batch_size, num_samples, num_feats + num_time_feats + num_cond_feats))
            assert out.shape[-1] == conditioning.output_dim
        else:
            out = conditioning(t, x)
            assert out.shape == ((batch_size, num_samples, num_feats + num_time_feats))
            assert out.shape[-1] == conditioning.output_dim

        # case 5
        t = jnp.zeros((batch_size, num_time_feats))
        cond = jnp.zeros((batch_size, num_cond_feats))
        x = jnp.zeros((batch_size, num_samples, num_feats))
        if concat_condition:
            out = conditioning(t, x, cond)
            assert out.shape == ((batch_size, num_samples, num_feats + num_time_feats + num_cond_feats))
            assert out.shape[-1] == conditioning.output_dim
        else:
            out = conditioning(t, x)
            assert out.shape == ((batch_size, num_samples, num_feats + num_time_feats))
            assert out.shape[-1] == conditioning.output_dim

        # case 6
        t = jnp.zeros((batch_size, num_samples, num_time_feats))
        cond = jnp.zeros((batch_size, num_samples, num_cond_feats))
        x = jnp.zeros((batch_size, num_samples, num_feats))
        if concat_condition:
            out = conditioning(t, x, cond)
            assert out.shape == ((batch_size, num_samples, num_feats + num_time_feats + num_cond_feats))
            assert out.shape[-1] == conditioning.output_dim
        else:
            out = conditioning(t, x)
            assert out.shape == ((batch_size, num_samples, num_feats + num_time_feats))
            assert out.shape[-1] == conditioning.output_dim

    @pytest.mark.parametrize("concat_condition", [True, False])
    def test_resnet1d_conditioning(
        self,
        concat_condition: bool
    ) -> None:
        
        def create_and_initialize_resnet(
                concat_condition: bool,
                t: ArrayLike,
                cond: ArrayLike,
                x: ArrayLike,
        ) -> tuple[Resnet1dConditioning, dict]:
            rng = jax.random.PRNGKey(0)

            if concat_condition:
                conditioning = Resnet1dConditioning(
                    num_feats,
                    num_time_feats,
                    num_cond_feats,
                )
                params = conditioning.init(rng, t, x, cond)
            else:
                conditioning = Resnet1dConditioning(
                    num_feats,
                    num_time_feats,
                )
                params = conditioning.init(rng, t, x)
            return conditioning, params

        # case 0
        t = jnp.zeros((num_time_feats, ))
        cond = jnp.zeros((num_cond_feats, ))
        x = jnp.zeros((batch_size, num_feats))
        conditioning, params = create_and_initialize_resnet(concat_condition, t, cond, x)
        if concat_condition:
            out = conditioning.apply(params, t, x, cond)
            assert out.shape == ((batch_size, num_feats))
            assert out.shape[-1] == conditioning.output_dim
        else:
            out = conditioning.apply(params, t, x)
            assert out.shape == ((batch_size, num_feats))
            assert out.shape[-1] == conditioning.output_dim

        # case 1
        t = jnp.zeros((1, num_time_feats))
        cond = jnp.zeros((1, num_cond_feats))
        x = jnp.zeros((batch_size, num_feats))
        conditioning, params = create_and_initialize_resnet(concat_condition, t, cond, x)
        if concat_condition:
            out = conditioning.apply(params, t, x, cond)
            assert out.shape == ((batch_size, num_feats))
            assert out.shape[-1] == conditioning.output_dim
        else:
            out = conditioning.apply(params, t, x)
            assert out.shape == ((batch_size, num_feats))
            assert out.shape[-1] == conditioning.output_dim

        # case 2
        t = jnp.zeros((batch_size, num_time_feats))
        cond = jnp.zeros((batch_size, num_cond_feats))
        x = jnp.zeros((batch_size, num_feats))
        conditioning, params = create_and_initialize_resnet(concat_condition, t, cond, x)
        if concat_condition:
            out = conditioning.apply(params, t, x, cond)
            assert out.shape == ((batch_size, num_feats))
            assert out.shape[-1] == conditioning.output_dim
        else:
            out = conditioning.apply(params, t, x)
            assert out.shape == ((batch_size, num_feats))
            assert out.shape[-1] == conditioning.output_dim

        # case 3
        t = jnp.zeros((num_time_feats, ))
        cond = jnp.zeros((num_cond_feats, ))
        x = jnp.zeros((batch_size, num_samples, num_feats))
        conditioning, params = create_and_initialize_resnet(concat_condition, t, cond, x)
        if concat_condition:
            out = conditioning.apply(params, t, x, cond)
            assert out.shape == ((batch_size, num_samples, num_feats))
            assert out.shape[-1] == conditioning.output_dim
        else:
            out = conditioning.apply(params, t, x)
            assert out.shape == ((batch_size, num_samples, num_feats))
            assert out.shape[-1] == conditioning.output_dim

        # case 4
        t = jnp.zeros((1, num_time_feats))
        cond = jnp.zeros((1, num_cond_feats))
        x = jnp.zeros((batch_size, num_samples, num_feats))
        conditioning, params = create_and_initialize_resnet(concat_condition, t, cond, x)
        if concat_condition:
            out = conditioning.apply(params, t, x, cond)
            assert out.shape == ((batch_size, num_samples, num_feats))
            assert out.shape[-1] == conditioning.output_dim
        else:
            out = conditioning.apply(params, t, x)
            assert out.shape == ((batch_size, num_samples, num_feats))
            assert out.shape[-1] == conditioning.output_dim

        # case 5
        t = jnp.zeros((batch_size, num_time_feats))
        cond = jnp.zeros((batch_size, num_cond_feats))
        x = jnp.zeros((batch_size, num_samples, num_feats))
        conditioning, params = create_and_initialize_resnet(concat_condition, t, cond, x)
        if concat_condition:
            out = conditioning.apply(params, t, x, cond)
            assert out.shape == ((batch_size, num_samples, num_feats))
            assert out.shape[-1] == conditioning.output_dim
        else:
            out = conditioning.apply(params, t, x)
            assert out.shape == ((batch_size, num_samples, num_feats))
            assert out.shape[-1] == conditioning.output_dim

        # case 6
        t = jnp.zeros((batch_size, num_samples, num_time_feats))
        cond = jnp.zeros((batch_size, num_samples, num_cond_feats))
        x = jnp.zeros((batch_size, num_samples, num_feats))
        conditioning, params = create_and_initialize_resnet(concat_condition, t, cond, x)
        if concat_condition:
            out = conditioning.apply(params, t, x, cond)
            assert out.shape == ((batch_size, num_samples, num_feats))
            assert out.shape[-1] == conditioning.output_dim
        else:
            out = conditioning.apply(params, t, x)
            assert out.shape == ((batch_size, num_samples, num_feats))
            assert out.shape[-1] == conditioning.output_dim

    def test_make_custom_conditioning_layer(
        self,
    ) -> None:
        
        ...

    def test_get_conditioning_layer(
        self,
    ) -> None:

        ...