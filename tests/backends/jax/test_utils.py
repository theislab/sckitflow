import jax.numpy as jnp
import pytest

from sc_flow._runtime import set_backend
from sc_flow.backends.jax._utils import broadcast_to_target_shape


class TestTorchBackendUtils:
    def test_broadcast_to_target_shape(
        self,
    ) -> None:
        set_backend("jax")

        # defining variables
        batch_size = 8
        num_feats = 16
        num_channels = 3
        height = 32
        width = 64

        # defining target shapes for each case
        target_shape2d = (batch_size, num_feats)
        target_shape3d = (batch_size, num_channels, height, width)

        # target 2D-case 0: same shape
        input_array = jnp.zeros(target_shape2d)
        input_array = broadcast_to_target_shape(input_array, target_shape2d)
        assert input_array.shape == target_shape2d

        # target 2D-case 1: same ndim with singleton trailing dim
        input_array = jnp.zeros((batch_size, 1))
        input_array = broadcast_to_target_shape(input_array, target_shape2d)
        assert input_array.shape == target_shape2d

        # target 2D-case 2: only batch dim
        input_array = jnp.zeros((batch_size,))
        input_array = broadcast_to_target_shape(input_array, target_shape2d)
        assert input_array.shape == target_shape2d

        # target 2D-case 3: only batch dim with singleton
        input_array = jnp.zeros((1,))
        input_array = broadcast_to_target_shape(input_array, target_shape2d)
        assert input_array.shape == target_shape2d

        # target 2D-case 4: wrong batch dimension
        with pytest.raises(ValueError, match=r"Mismatch in shape"):
            input_array = jnp.zeros((batch_size + 1,))
            input_array = broadcast_to_target_shape(input_array, target_shape2d)
            return None

        # target 2D-case5: more ndim in input than in target shape
        with pytest.raises(ValueError, match=r"more dimensions"):
            input_array = jnp.zeros((batch_size, 1, 1))
            input_array = broadcast_to_target_shape(input_array, target_shape2d)
            return None

        # target3D-case0: same shape
        input_array = jnp.zeros(target_shape3d)
        input_array = broadcast_to_target_shape(input_array, target_shape3d)
        assert input_array.shape == target_shape3d

        # target 3D-case 1: same batch size with one singleton trailing dim
        input_array = jnp.zeros(
            (
                batch_size,
                1,
            )
        )
        input_array = broadcast_to_target_shape(input_array, target_shape3d)
        assert input_array.shape == target_shape3d

        # target 3D-case 2: same batch size with two singleton trailing dim
        input_array = jnp.zeros((batch_size, 1, 1))
        input_array = broadcast_to_target_shape(input_array, target_shape3d)
        assert input_array.shape == target_shape3d

        # target 3D-case 3: same batch size with three singleton trailing dim
        input_array = jnp.zeros(
            (
                batch_size,
                1,
                1,
                1,
            )
        )
        input_array = broadcast_to_target_shape(input_array, target_shape3d)
        assert input_array.shape == target_shape3d

        # target 3D-case 4: only batch dim
        input_array = jnp.zeros((batch_size,))
        input_array = broadcast_to_target_shape(input_array, target_shape3d)
        assert input_array.shape == target_shape3d

        # target 3D-case 5: only batch dim with singleton
        input_array = jnp.zeros((1,))
        input_array = broadcast_to_target_shape(input_array, target_shape3d)
        assert input_array.shape == target_shape3d

        # target 3D-case 6: wrong batch dimension
        with pytest.raises(ValueError, match=r"Mismatch in shape"):
            input_array = jnp.zeros((batch_size + 1,))
            input_array = broadcast_to_target_shape(input_array, target_shape3d)
            return None

        # target 3D-case 7: more ndim in input than in target shape
        with pytest.raises(ValueError, match=r"more dimensions"):
            input_array = jnp.zeros((batch_size, 1, 1, 1, 1))
            input_array = broadcast_to_target_shape(input_array, target_shape3d)
            return None

    def test_make_concatenation_possible(
        self,
    ) -> None:
        pass
