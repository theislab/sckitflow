import pytest
import torch

from sc_flow.flow._torch_utils import (
    broadcast_to_target_shape,
    ensure_2d_tensor_with_singleton_trailing_dim,
    make_concatenation_possible,
)

# defining variables
batch_size = 8
num_samples = 50
num_feats = 16
num_time_feats = 6
num_channels = 3
height = 32
width = 64


class TestTorchBackendUtils:
    def test_broadcast_to_target_shape(
        self,
    ) -> None:
        # defining target shapes for each case
        target_shape2d = (batch_size, num_feats)
        target_shape3d = (batch_size, num_channels, height, width)

        # target 2D-case 0: same shape
        input_tensor = torch.zeros(target_shape2d)
        input_tensor = broadcast_to_target_shape(input_tensor, target_shape2d)
        assert input_tensor.shape == target_shape2d

        # target 2D-case 1: same ndim with singleton trailing dim
        input_tensor = torch.zeros((batch_size, 1))
        input_tensor = broadcast_to_target_shape(input_tensor, target_shape2d)
        assert input_tensor.shape == target_shape2d

        # target 2D-case 2: only batch dim
        input_tensor = torch.zeros((batch_size,))
        input_tensor = broadcast_to_target_shape(input_tensor, target_shape2d)
        assert input_tensor.shape == target_shape2d

        # target 2D-case 3: only batch dim with singleton
        input_tensor = torch.zeros((1,))
        input_tensor = broadcast_to_target_shape(input_tensor, target_shape2d)
        assert input_tensor.shape == target_shape2d

        # target 2D-case 4: wrong batch dimension
        with pytest.raises(ValueError, match=r"Mismatch in shape"):
            input_tensor = torch.zeros((batch_size + 1,))
            input_tensor = broadcast_to_target_shape(input_tensor, target_shape2d)
            return None

        # target 2D-case5: more ndim in input than in target shape
        with pytest.raises(ValueError, match=r"more dimensions"):
            input_tensor = torch.zeros((batch_size, 1, 1))
            input_tensor = broadcast_to_target_shape(input_tensor, target_shape2d)
            return None

        # target3D-case0: same shape
        input_tensor = torch.zeros(target_shape3d)
        input_tensor = broadcast_to_target_shape(input_tensor, target_shape3d)
        assert input_tensor.shape == target_shape3d

        # target 3D-case 1: same batch size with one singleton trailing dim
        input_tensor = torch.zeros(
            (
                batch_size,
                1,
            )
        )
        input_tensor = broadcast_to_target_shape(input_tensor, target_shape3d)
        assert input_tensor.shape == target_shape3d

        # target 3D-case 2: same batch size with two singleton trailing dim
        input_tensor = torch.zeros((batch_size, 1, 1))
        input_tensor = broadcast_to_target_shape(input_tensor, target_shape3d)
        assert input_tensor.shape == target_shape3d

        # target 3D-case 3: same batch size with three singleton trailing dim
        input_tensor = torch.zeros(
            (
                batch_size,
                1,
                1,
                1,
            )
        )
        input_tensor = broadcast_to_target_shape(input_tensor, target_shape3d)
        assert input_tensor.shape == target_shape3d

        # target 3D-case 4: only batch dim
        input_tensor = torch.zeros((batch_size,))
        input_tensor = broadcast_to_target_shape(input_tensor, target_shape3d)
        assert input_tensor.shape == target_shape3d

        # target 3D-case 5: only batch dim with singleton
        input_tensor = torch.zeros((1,))
        input_tensor = broadcast_to_target_shape(input_tensor, target_shape3d)
        assert input_tensor.shape == target_shape3d

        # target 3D-case 6: wrong batch dimension
        with pytest.raises(ValueError, match=r"Mismatch in shape"):
            input_tensor = torch.zeros((batch_size + 1,))
            input_tensor = broadcast_to_target_shape(input_tensor, target_shape3d)
            return None

        # target 3D-case 7: more ndim in input than in target shape
        with pytest.raises(ValueError, match=r"more dimensions"):
            input_tensor = torch.zeros((batch_size, 1, 1, 1, 1))
            input_tensor = broadcast_to_target_shape(input_tensor, target_shape3d)
            return None

    def test_ensure_2d_tensor_with_singleton_trailing_dim(
        self,
    ) -> None:
        # case 0: t: () x: (B, D)
        t = torch.zeros(())
        t = ensure_2d_tensor_with_singleton_trailing_dim(t)
        assert t.shape == (1, 1)

        # case 0: t: () x: (B, D)
        t = torch.zeros((batch_size,))
        t = ensure_2d_tensor_with_singleton_trailing_dim(t)
        assert t.shape == (batch_size, 1)

        # case 0: t: () x: (B, D)
        t = torch.zeros((batch_size, 1))
        t = ensure_2d_tensor_with_singleton_trailing_dim(t)
        assert t.shape == (batch_size, 1)

    def test_make_concatenation_on_trailing_dim_possible(
        self,
    ) -> None:
        # case 0: t: () x: (B, D)
        t = torch.zeros(())
        x = torch.zeros((batch_size, num_feats))
        t = make_concatenation_possible(t, x, -1)
        assert t.shape == ((batch_size,))

        # case 1: t: (num_feats, ) x: (B, D)
        t = torch.zeros((num_time_feats,))
        x = torch.zeros((batch_size, num_feats))
        t = make_concatenation_possible(t, x, -1)
        assert t.shape == ((batch_size, num_time_feats))

        # case 1: t: (B, 1) x: (B, D)
        t = torch.zeros((batch_size, num_time_feats))
        x = torch.zeros((batch_size, num_feats))
        t = make_concatenation_possible(t, x, -1)
        assert t.shape == ((batch_size, num_time_feats))

        # case 0: t: () x: (B, D)
        t = torch.zeros(())
        x = torch.zeros((batch_size, num_samples, num_feats))
        t = make_concatenation_possible(t, x, -1)
        assert t.shape == (
            (
                batch_size,
                num_samples,
            )
        )

        # case 1: t: (num_feats, ) x: (B, D)
        t = torch.zeros((num_time_feats,))
        x = torch.zeros((batch_size, num_samples, num_feats))
        t = make_concatenation_possible(t, x, -1)
        assert t.shape == ((batch_size, num_samples, num_time_feats))

        # case 1: t: (B, 1) x: (B, D)
        t = torch.zeros((batch_size, num_time_feats))
        x = torch.zeros((batch_size, num_samples, num_feats))
        t = make_concatenation_possible(t, x, -1)
        assert t.shape == ((batch_size, num_samples, num_time_feats))

        # case 1: t: (B, 1) x: (B, D)
        t = torch.zeros((batch_size, num_samples, num_time_feats))
        x = torch.zeros((batch_size, num_samples, num_feats))
        t = make_concatenation_possible(t, x, -1)
        assert t.shape == ((batch_size, num_samples, num_time_feats))
