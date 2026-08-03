from collections.abc import Sequence
from typing import Any

import pytest
import torch

from sckitflow.core.nn._modules import MLP, Resnet1d

output_dim = 32

# Normalization / regularization flags are handled independently of one another by
# `MLP._make_layer` and `Resnet1d`, so the full cartesian product of the nine boolean
# axes (512 combinations) adds no coverage over a set that exercises every value of
# every flag plus the interactions that actually share code paths: batchnorm x
# layernorm x dropout x bias.
LAYER_CONFIGS = [
    pytest.param({}, id="plain"),
    pytest.param({"bias": False}, id="nobias"),
    pytest.param({"dropout_p": 0.1}, id="dropout"),
    pytest.param({"dropout_p": 0.1, "dropout_inplace": True, "bias": False}, id="dropout-inplace"),
    pytest.param(
        {"use_batchnorm": True, "batchnorm_affine": True, "batchnorm_track_running_stats": True},
        id="bn",
    ),
    pytest.param(
        {"use_batchnorm": True, "batchnorm_affine": False, "batchnorm_track_running_stats": True, "bias": False},
        id="bn-noaffine",
    ),
    pytest.param(
        {
            "use_batchnorm": True,
            "batchnorm_affine": True,
            "batchnorm_track_running_stats": False,
            "dropout_p": 0.1,
            "dropout_inplace": True,
        },
        id="bn-notrack",
    ),
    pytest.param(
        {
            "use_batchnorm": True,
            "batchnorm_affine": False,
            "batchnorm_track_running_stats": False,
            "dropout_p": 0.1,
            "bias": False,
        },
        id="bn-noaffine-notrack",
    ),
    pytest.param(
        {"use_layernorm": True, "layernorm_elementwise_affine": True, "layernorm_bias": True},
        id="ln",
    ),
    pytest.param(
        {
            "use_layernorm": True,
            "layernorm_elementwise_affine": True,
            "layernorm_bias": False,
            "dropout_p": 0.1,
        },
        id="ln-nobias",
    ),
    pytest.param(
        {
            "use_layernorm": True,
            "layernorm_elementwise_affine": False,
            "layernorm_bias": False,
            "bias": False,
        },
        id="ln-noaffine",
    ),
    pytest.param(
        {
            "use_batchnorm": True,
            "batchnorm_affine": True,
            "batchnorm_track_running_stats": True,
            "use_layernorm": True,
            "layernorm_elementwise_affine": True,
            "layernorm_bias": True,
            "dropout_p": 0.1,
        },
        id="bn-ln-dropout",
    ),
    pytest.param(
        {
            "use_batchnorm": True,
            "batchnorm_affine": False,
            "batchnorm_track_running_stats": False,
            "use_layernorm": True,
            "layernorm_elementwise_affine": False,
            "layernorm_bias": False,
            "dropout_p": 0.1,
            "dropout_inplace": True,
            "bias": False,
        },
        id="bn-ln-dropout-nobias",
    ),
]


class TestNNModules:
    _input_dim: int = 2
    _output_dim: int = 5
    _batch_size: int = 8
    _embedding_dim: int = 16

    @pytest.mark.parametrize("hidden_dims", [None, (), (16,), (16, 16)])
    @pytest.mark.parametrize("layer_config", LAYER_CONFIGS)
    def test_mlp(
        self,
        hidden_dims: Sequence[int],
        layer_config: dict[str, Any],
    ) -> None:
        use_batchnorm = layer_config.get("use_batchnorm", False)
        mlp = MLP(
            self._input_dim,
            self._output_dim,
            hidden_dims=hidden_dims,
            **layer_config,
        )

        # case 0: (1, D)
        input_tensor = torch.zeros((1, self._input_dim))
        if use_batchnorm:
            with pytest.raises(ValueError, match=r"more than 1 value per channel"):
                output_tensor = mlp(input_tensor)
        else:
            output_tensor = mlp(input_tensor)
            assert output_tensor.shape == (1, self._output_dim)

        # case 1: (D)
        input_tensor = torch.zeros(self._input_dim)
        if use_batchnorm:
            with pytest.raises(ValueError, match=r".*"):
                output_tensor = mlp(input_tensor)
                return None
        else:
            output_tensor = mlp(input_tensor)
            assert output_tensor.shape == (self._output_dim,)

        # case 2: (B, D)
        input_tensor = torch.zeros((self._batch_size, self._input_dim))
        output_tensor = mlp(input_tensor)
        assert output_tensor.shape == (self._batch_size, self._output_dim)

        # case 3: (1, B, D)
        input_tensor = torch.zeros((1, self._batch_size, self._input_dim))
        output_tensor = mlp(input_tensor)
        assert output_tensor.shape == (1, self._batch_size, self._output_dim)

        # case 4: (B, D + 1) Error
        input_tensor = torch.zeros((self._batch_size, self._input_dim + 1))
        with pytest.raises(
            RuntimeError,
            match=r"shapes cannot be multiplied",
        ):
            output_tensor = mlp(input_tensor)
            return None

    @pytest.mark.parametrize("activation_cls", [None, torch.nn.ReLU])
    @pytest.mark.parametrize("final_activation_cls", [None, torch.nn.Identity])
    @pytest.mark.parametrize("activation_cls_kwargs", [None, {}])
    @pytest.mark.parametrize("final_activation_cls_kwargs", [None, {}])
    def test_mlp_activations(
        self,
        activation_cls: type[torch.nn.Module] | None,
        final_activation_cls: type[torch.nn.Module] | None,
        activation_cls_kwargs: dict[str, Any] | None,
        final_activation_cls_kwargs: dict[str, Any] | None,
    ) -> None:
        mlp = MLP(
            self._input_dim,
            self._output_dim,
            hidden_dims=(16, 16),
            activation_cls=activation_cls,
            final_activation_cls=final_activation_cls,
            activation_cls_kwargs=activation_cls_kwargs,
            final_activation_cls_kwargs=final_activation_cls_kwargs,
        )

        input_tensor = torch.zeros((self._batch_size, self._input_dim))
        output_tensor = mlp(input_tensor)
        assert output_tensor.shape == (self._batch_size, self._output_dim)

    @pytest.mark.parametrize("num_resnet_layers", [None, 5])
    @pytest.mark.parametrize("output_dim", [None, output_dim])
    @pytest.mark.parametrize("layer_config", LAYER_CONFIGS)
    def test_resnet1d(
        self,
        num_resnet_layers: int,
        output_dim: int | None,
        layer_config: dict[str, Any],
    ):
        use_batchnorm = layer_config.get("use_batchnorm", False)
        resnet = Resnet1d(
            self._input_dim,
            self._embedding_dim,
            num_resnet_layers,
            output_dim=output_dim,
            **layer_config,
        )

        # case 0: x.shape = (1, D), cond.shape = (1, K)
        input_tensor = torch.zeros((1, self._input_dim))
        condition = torch.zeros((1, self._embedding_dim))
        if use_batchnorm:
            with pytest.raises(ValueError, match=r"more than 1 value per channel"):
                output_tensor = resnet(input_tensor, condition)
        else:
            output_tensor = resnet(input_tensor, condition)
            if output_dim is None:
                assert output_tensor.shape == (1, self._input_dim)
            else:
                assert output_tensor.shape == (1, output_dim)

        # case 1: x.shape = (D), cond.shape = (K)
        input_tensor = torch.zeros((self._input_dim,))
        condition = torch.zeros((self._embedding_dim,))
        if use_batchnorm:
            with pytest.raises(ValueError, match=r".*"):
                output_tensor = resnet(input_tensor, condition)
                return None
        else:
            output_tensor = resnet(input_tensor, condition)
            if output_dim is None:
                assert output_tensor.shape == (self._input_dim,)
            else:
                assert output_tensor.shape == (output_dim,)

        # case 2: x.shape = (B, D), cond.shape = (B, K)
        input_tensor = torch.zeros((self._batch_size, self._input_dim))
        condition = torch.zeros((self._batch_size, self._embedding_dim))
        output_tensor = resnet(input_tensor, condition)
        if output_dim is None:
            assert output_tensor.shape == (self._batch_size, self._input_dim)
        else:
            assert output_tensor.shape == (self._batch_size, output_dim)

        # case 3: (1, B, D)
        input_tensor = torch.zeros((1, self._batch_size, self._input_dim))
        condition = torch.zeros((self._batch_size, self._embedding_dim))
        output_tensor = resnet(input_tensor, condition)
        if output_dim is None:
            assert output_tensor.shape == (1, self._batch_size, self._input_dim)
        else:
            assert output_tensor.shape == (1, self._batch_size, output_dim)

        # case 4: x.shape = (B, D + 1), cond.shape = (B, K)
        input_tensor = torch.zeros((self._batch_size, self._input_dim + 1))
        condition = torch.zeros((self._batch_size, self._embedding_dim))
        if use_batchnorm:
            with pytest.raises(RuntimeError, match=r"(cannot be multiplied)|(should contain)|(Given normalized_shape)"):
                output_tensor = resnet(input_tensor, condition)
                return None
        else:
            with pytest.raises(
                RuntimeError,
                match=r"(cannot be multiplied)|(Given normalized_shape)",
            ):
                output_tensor = resnet(input_tensor, condition)
                return None

        # case 5: x.shape = (B, D), cond.shape = (B + 1, K)
        input_tensor = torch.zeros((self._batch_size, self._input_dim))
        condition = torch.zeros((self._batch_size + 1, self._embedding_dim))
        with pytest.raises(
            RuntimeError,
            # match=r"Shape mismatch between hidden condition and state projection\."
        ):
            output_tensor = resnet(input_tensor, condition)
            return None

    @pytest.mark.parametrize("activation_cls", [None, torch.nn.SiLU])
    @pytest.mark.parametrize("activation_cls_kwargs", [None, {}])
    def test_resnet1d_activations(
        self,
        activation_cls: type[torch.nn.Module] | None,
        activation_cls_kwargs: dict[str, Any] | None,
    ):
        resnet = Resnet1d(
            self._input_dim,
            self._embedding_dim,
            5,
            activation_cls=activation_cls,
            activation_cls_kwargs=activation_cls_kwargs,
        )

        input_tensor = torch.zeros((self._batch_size, self._input_dim))
        condition = torch.zeros((self._batch_size, self._embedding_dim))
        output_tensor = resnet(input_tensor, condition)
        assert output_tensor.shape == (self._batch_size, self._input_dim)
