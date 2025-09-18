from collections.abc import Sequence
from typing import Any

import pytest
import torch

from sc_flow.backends.torch.nn._modules import MLP, Resnet1d

output_dim = 32


class TestNNModules:
    _input_dim: int = 2
    _output_dim: int = 5
    _batch_size: int = 8
    _embedding_dim: int = 16

    @pytest.mark.parametrize("hidden_dims", [None, (), (16,), (16, 16)])
    @pytest.mark.parametrize("activation_cls", [None, torch.nn.ReLU])
    @pytest.mark.parametrize("final_activation_cls", [None, torch.nn.Identity])
    @pytest.mark.parametrize("use_batchnorm", [True, False])
    @pytest.mark.parametrize("batchnorm_eps", [1e-3])
    @pytest.mark.parametrize("batchnorm_momentum", [1e-2])
    @pytest.mark.parametrize("batchnorm_affine", [True, False])
    @pytest.mark.parametrize("batchnorm_track_running_stats", [True, False])
    @pytest.mark.parametrize("use_layernorm", [True, False])
    @pytest.mark.parametrize("layernorm_eps", [1e-5])
    @pytest.mark.parametrize("layernorm_elementwise_affine", [True, False])
    @pytest.mark.parametrize("layernorm_bias", [True, False])
    @pytest.mark.parametrize("dropout_p", [0.0, 0.1])
    @pytest.mark.parametrize("dropout_inplace", [True, False])
    @pytest.mark.parametrize("activation_cls_kwargs", [None, {}])
    @pytest.mark.parametrize("final_activation_cls_kwargs", [None, {}])
    @pytest.mark.parametrize("bias", [True, False])
    def test_mlp(
        self,
        hidden_dims: Sequence[int],
        activation_cls: type[torch.nn.Module],
        final_activation_cls: type[torch.nn.Module],
        use_batchnorm: bool,
        batchnorm_eps: float,
        batchnorm_momentum: float,
        batchnorm_affine: bool,
        batchnorm_track_running_stats: bool,
        use_layernorm: bool,
        layernorm_eps: float,
        layernorm_elementwise_affine: bool,
        layernorm_bias: bool,
        dropout_p: float,
        dropout_inplace: bool,
        activation_cls_kwargs: dict[str, Any] | None,
        final_activation_cls_kwargs: dict[str, Any] | None,
        bias: bool,
    ) -> None:
        mlp = MLP(
            self._input_dim,
            self._output_dim,
            hidden_dims=hidden_dims,
            activation_cls=activation_cls,
            final_activation_cls=final_activation_cls,
            use_batchnorm=use_batchnorm,
            batchnorm_eps=batchnorm_eps,
            batchnorm_momentum=batchnorm_momentum,
            batchnorm_affine=batchnorm_affine,
            batchnorm_track_running_stats=batchnorm_track_running_stats,
            use_layernorm=use_layernorm,
            layernorm_eps=layernorm_eps,
            layernorm_elementwise_affine=layernorm_elementwise_affine,
            layernorm_bias=layernorm_bias,
            dropout_p=dropout_p,
            dropout_inplace=dropout_inplace,
            activation_cls_kwargs=activation_cls_kwargs,
            final_activation_cls_kwargs=final_activation_cls_kwargs,
            bias=bias,
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
        if use_batchnorm and (batchnorm_track_running_stats or batchnorm_affine):
            # if use_batchnorm:
            with pytest.raises(RuntimeError, match=r"should contain"):
                output_tensor = mlp(input_tensor)
                return None
        else:
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

    @pytest.mark.parametrize("num_resnet_layers", [1, 3])
    @pytest.mark.parametrize("output_dim", [None, output_dim])
    @pytest.mark.parametrize("activation_cls", [None, torch.nn.SiLU])
    @pytest.mark.parametrize("use_batchnorm", [True, False])
    @pytest.mark.parametrize("batchnorm_eps", [1e-3])
    @pytest.mark.parametrize("batchnorm_momentum", [1e-2])
    @pytest.mark.parametrize("batchnorm_affine", [True, False])
    @pytest.mark.parametrize("batchnorm_track_running_stats", [True, False])
    @pytest.mark.parametrize("use_layernorm", [True, False])
    @pytest.mark.parametrize("layernorm_eps", [1e-5])
    @pytest.mark.parametrize("layernorm_elementwise_affine", [True, False])
    @pytest.mark.parametrize("layernorm_bias", [True, False])
    @pytest.mark.parametrize("dropout_p", [0.0, 0.1])
    @pytest.mark.parametrize("dropout_inplace", [True, False])
    @pytest.mark.parametrize("activation_cls_kwargs", [None, {}])
    @pytest.mark.parametrize("bias", [True, False])
    def test_resnet1d(
        self,
        num_resnet_layers: int,
        output_dim: int | None,
        activation_cls: torch.nn.Module | None,
        use_batchnorm: bool,
        batchnorm_eps: float,
        batchnorm_momentum: float,
        batchnorm_affine: bool,
        batchnorm_track_running_stats: bool,
        use_layernorm: bool,
        layernorm_eps: float,
        layernorm_elementwise_affine: bool,
        layernorm_bias: bool,
        dropout_p: float,
        dropout_inplace: bool,
        activation_cls_kwargs: dict[str, Any] | None,
        bias: bool,
    ):
        resnet = Resnet1d(
            self._input_dim,
            num_resnet_layers,
            output_dim=output_dim,
            activation_cls=activation_cls,
            embedding_dim=self._embedding_dim,
            use_batchnorm=use_batchnorm,
            batchnorm_eps=batchnorm_eps,
            batchnorm_momentum=batchnorm_momentum,
            batchnorm_affine=batchnorm_affine,
            batchnorm_track_running_stats=batchnorm_track_running_stats,
            use_layernorm=use_layernorm,
            layernorm_eps=layernorm_eps,
            layernorm_elementwise_affine=layernorm_elementwise_affine,
            layernorm_bias=layernorm_bias,
            dropout_p=dropout_p,
            dropout_inplace=dropout_inplace,
            activation_cls_kwargs=activation_cls_kwargs,
            bias=bias,
        )
        print(resnet)

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
                assert output_tensor.shape == (self._input_dim, )
            else:
                assert output_tensor.shape == (output_dim, )

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
        if use_batchnorm and (batchnorm_track_running_stats or batchnorm_affine):
            with pytest.raises(RuntimeError, match=r"should contain"):
                output_tensor = resnet(input_tensor, condition)
                return None
        else:
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
            #match=r"Shape mismatch between hidden condition and state projection\."
        ):
            output_tensor = resnet(input_tensor, condition)
            return None
