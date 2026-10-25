"""Tests for :class:`ResNet` and :class:`ResNetConfig`.

The suite intentionally covers two layers of failure:

* the residual/conditioning math of the module itself (shape contracts,
  residual identity, condition sensitivity, parameter gradients), and
* the config wrapper's ``build`` glue (defaults, field forwarding, and the
  ``self.dict()`` / ``type``-leak regression).

Batchnorm rejection of batch size 1 / unbatched input is train-mode-only and
is covered by two dedicated tests; the layer-config matrix runs in ``.eval()``
so dropout is deterministic and does **not** re-check the batchnorm error.

Import path: adjust ``sckitflow.core.nn`` to wherever ``ResNet`` /
``ResNetConfig`` actually live.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from sckitflow.core.nn.blocks._resnet import ResNet, ResNetConfig

# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

_INPUT_DIM = 8
_EMBEDDING_DIM = 16
_BATCH_SIZE = 4

# Same rationale as the MLP suite: layer-construction axes are independent
# within ``_make_block``, so a targeted subset covers all code paths without
# enumerating the full cartesian product.
LAYER_CONFIGS = [
    pytest.param({}, id="plain"),
    pytest.param({"bias": False}, id="nobias"),
    pytest.param({"dropout_p": 0.1}, id="dropout"),
    pytest.param(
        {"dropout_p": 0.1, "dropout_inplace": True, "bias": False},
        id="dropout-inplace",
    ),
    pytest.param(
        {"use_batchnorm": True, "batchnorm_affine": True, "batchnorm_track_running_stats": True},
        id="bn",
    ),
    pytest.param(
        {
            "use_batchnorm": True,
            "batchnorm_affine": False,
            "batchnorm_track_running_stats": False,
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
            "layernorm_elementwise_affine": False,
            "layernorm_bias": False,
            "bias": False,
        },
        id="ln-noaffine",
    ),
    pytest.param(
        {
            "use_batchnorm": True,
            "use_layernorm": True,
            "dropout_p": 0.1,
            "bias": False,
        },
        id="bn-ln-dropout-nobias",
    ),
]

OUTPUT_DIMS = [None, 4]
NUM_LAYERS = [1, 3]

INPUT_SHAPES = [
    pytest.param((_BATCH_SIZE, _INPUT_DIM), (_BATCH_SIZE, _EMBEDDING_DIM), id="batched"),
    pytest.param((_INPUT_DIM,), (_EMBEDDING_DIM,), id="unbatched"),
    pytest.param(
        (2, _BATCH_SIZE, _INPUT_DIM),
        (2, _BATCH_SIZE, _EMBEDDING_DIM),
        id="extra-lead-dim",
    ),
]


# ---------------------------------------------------------------------------
# ResNet
# ---------------------------------------------------------------------------


class TestResNet:
    _input_dim = _INPUT_DIM
    _embedding_dim = _EMBEDDING_DIM
    _batch_size = _BATCH_SIZE

    # ---- construction -----------------------------------------------------

    def test_output_dim_property_defaults_to_input_dim(self) -> None:
        mod = ResNet(self._input_dim, self._embedding_dim, num_resnet_layers=2)
        assert mod.output_dim == self._input_dim

    def test_output_dim_property_explicit(self) -> None:
        mod = ResNet(self._input_dim, self._embedding_dim, num_resnet_layers=2, output_dim=4)
        assert mod.output_dim == 4

    def test_num_layers_default_is_positive(self) -> None:
        mod = ResNet(self._input_dim, self._embedding_dim)  # relies on DEFAULT_NUM_RESNET_LAYERS
        assert mod._num_resnet_layers > 0
        assert len(list(mod._ResNet)) == mod._num_resnet_layers

    def test_num_layers_explicit(self) -> None:
        mod = ResNet(self._input_dim, self._embedding_dim, num_resnet_layers=5)
        assert mod._num_resnet_layers == 5
        assert len(list(mod._ResNet)) == 5

    # ---- forward shape contract ------------------------------------------

    @pytest.mark.parametrize("output_dim", OUTPUT_DIMS)
    @pytest.mark.parametrize("num_layers", NUM_LAYERS)
    @pytest.mark.parametrize("x_shape,cond_shape", INPUT_SHAPES)
    def test_forward_shapes(
        self,
        output_dim: int | None,
        num_layers: int,
        x_shape: tuple[int, ...],
        cond_shape: tuple[int, ...],
    ) -> None:
        mod = ResNet(
            self._input_dim,
            self._embedding_dim,
            num_resnet_layers=num_layers,
            output_dim=output_dim,
        ).eval()
        x = torch.randn(*x_shape)
        cond = torch.randn(*cond_shape)
        with torch.no_grad():
            y = mod(x, cond)
        expected = output_dim if output_dim is not None else self._input_dim
        assert y.shape == (*x_shape[:-1], expected)

    # ---- batchnorm constraints (train mode) ------------------------------

    def test_batchnorm_rejects_singleton_batch(self) -> None:
        mod = ResNet(self._input_dim, self._embedding_dim, num_resnet_layers=2, use_batchnorm=True)
        with pytest.raises(ValueError, match="more than 1 value per channel"):
            mod(torch.zeros(1, self._input_dim), torch.zeros(1, self._embedding_dim))

    def test_batchnorm_rejects_unbatched(self) -> None:
        mod = ResNet(self._input_dim, self._embedding_dim, num_resnet_layers=2, use_batchnorm=True)
        with pytest.raises(ValueError):
            mod(torch.zeros(self._input_dim), torch.zeros(self._embedding_dim))

    # ---- the residual property itself ------------------------------------

    def test_zeroed_hidden_branch_returns_input(self) -> None:
        """With every hidden-branch parameter zeroed, the residual update is
        ``x_new = skip(x) + 0 = x`` (``input_dim == output_dim`` so skip is
        ``Identity``). Defining property of a residual block.
        """
        mod = ResNet(self._input_dim, self._embedding_dim, num_resnet_layers=3).eval()
        x = torch.randn(self._batch_size, self._input_dim)
        cond = torch.randn(self._batch_size, self._embedding_dim)
        with torch.no_grad():
            for layer in mod._ResNet:
                layer_dict = dict(layer.named_children())
                for name in ("net1", "cond_proj", "net2"):
                    for p in layer_dict[name].parameters():
                        p.zero_()
            out = mod(x, cond)
        assert torch.allclose(out, x, atol=1e-6)

    def test_condition_changes_output(self) -> None:
        mod = ResNet(self._input_dim, self._embedding_dim, num_resnet_layers=2).eval()
        x = torch.randn(self._batch_size, self._input_dim)
        cond_a = torch.randn(self._batch_size, self._embedding_dim)
        cond_b = torch.randn(self._batch_size, self._embedding_dim)
        with torch.no_grad():
            assert not torch.allclose(mod(x, cond_a), mod(x, cond_b))

    def test_input_changes_output(self) -> None:
        mod = ResNet(self._input_dim, self._embedding_dim, num_resnet_layers=2).eval()
        cond = torch.randn(self._batch_size, self._embedding_dim)
        x_a = torch.randn(self._batch_size, self._input_dim)
        x_b = torch.randn(self._batch_size, self._input_dim)
        with torch.no_grad():
            assert not torch.allclose(mod(x_a, cond), mod(x_b, cond))

    def test_zero_layers_is_identity_and_ignores_output_dim(self) -> None:
        """``num_resnet_layers=0`` bypasses the ``output_dim`` projection;
        output stays ``input_dim``-shaped. Documents the edge case.
        """
        mod = ResNet(
            self._input_dim,
            self._embedding_dim,
            num_resnet_layers=0,
            output_dim=4,
        ).eval()
        x = torch.randn(self._batch_size, self._input_dim)
        cond = torch.randn(self._batch_size, self._embedding_dim)
        with torch.no_grad():
            out = mod(x, cond)
        assert out.shape == (self._batch_size, self._input_dim)  # not 4
        assert torch.allclose(out, x)

    # ---- layer-config matrix --------------------------------------------

    @pytest.mark.parametrize("layer_config", LAYER_CONFIGS)
    def test_forward_all_layer_configs(self, layer_config: dict[str, Any]) -> None:
        """Every layer-config combination produces the documented output shape.

        Runs in ``.eval()`` so dropout is deterministic. Batchnorm's
        train-mode rejection of batch size 1 is covered separately by
        ``test_batchnorm_rejects_singleton_batch``.
        """
        mod = ResNet(
            self._input_dim,
            self._embedding_dim,
            num_resnet_layers=2,
            **layer_config,
        ).eval()
        x = torch.randn(self._batch_size, self._input_dim)
        cond = torch.randn(self._batch_size, self._embedding_dim)
        with torch.no_grad():
            out = mod(x, cond)
        assert out.shape == (self._batch_size, self._input_dim)
        # NO batchnorm raises-check here — .eval() does not raise.

    @pytest.mark.parametrize("activation_cls", [None, torch.nn.ReLU, torch.nn.GELU])
    def test_various_activations(self, activation_cls: type[torch.nn.Module] | None) -> None:
        mod = ResNet(
            self._input_dim,
            self._embedding_dim,
            num_resnet_layers=2,
            activation_cls=activation_cls,
        ).eval()
        x = torch.randn(self._batch_size, self._input_dim)
        cond = torch.randn(self._batch_size, self._embedding_dim)
        with torch.no_grad():
            assert mod(x, cond).shape == (self._batch_size, self._input_dim)

    # ---- bias propagation (bug-catchers) --------------------------------

    def test_bias_false_propagates_to_skip_proj(self) -> None:
        """``bias=False`` must apply to *every* ``Linear``, including the skip
        projection when ``input_dim != output_dim``.

        Only the **first** layer has a ``Linear`` skip (input_dim != output_dim);
        subsequent layers keep ``input_dim == output_dim`` so skip is
        ``Identity`` and has no bias at all. The loop therefore checks
        Linear-shaped skips only.
        """
        mod = ResNet(
            self._input_dim,
            self._embedding_dim,
            num_resnet_layers=2,
            output_dim=4,  # forces a Linear skip_proj on layer 0
            bias=False,
        )
        saw_linear_skip = False
        for layer in mod._ResNet:
            skip = dict(layer.named_children())["skip_proj"]
            if isinstance(skip, torch.nn.Identity):
                continue
            assert isinstance(skip, torch.nn.Linear)
            assert skip.bias is None, "skip_proj must honour bias=False"
            saw_linear_skip = True
        assert saw_linear_skip, "expected at least one Linear skip_proj"

    def test_bias_false_propagates_to_all_other_linears(self) -> None:
        mod = ResNet(
            self._input_dim,
            self._embedding_dim,
            num_resnet_layers=2,
            output_dim=4,
            bias=False,
        )
        for layer in mod._ResNet:
            for name, sub in layer.named_children():
                if name == "skip_proj":
                    continue
                for m in sub.modules():
                    if isinstance(m, torch.nn.Linear):
                        assert m.bias is None

    # ---- gradients -------------------------------------------------------

    def test_all_parameters_receive_gradients(self) -> None:
        mod = ResNet(self._input_dim, self._embedding_dim, num_resnet_layers=2)
        x = torch.randn(self._batch_size, self._input_dim)
        cond = torch.randn(self._batch_size, self._embedding_dim)
        mod(x, cond).sum().backward()
        for name, p in mod.named_parameters():
            assert p.grad is not None, f"parameter {name!r} received no gradient"

    # ---- train/eval semantics --------------------------------------------

    def test_eval_mode_is_deterministic_with_dropout(self) -> None:
        torch.manual_seed(0)
        mod = ResNet(self._input_dim, self._embedding_dim, num_resnet_layers=2, dropout_p=0.5).eval()
        x = torch.randn(self._batch_size, self._input_dim)
        cond = torch.randn(self._batch_size, self._embedding_dim)
        with torch.no_grad():
            assert torch.allclose(mod(x, cond), mod(x, cond))


# ---------------------------------------------------------------------------
# ResNetConfig
# ---------------------------------------------------------------------------


class TestResNetConfig:
    _input_dim = _INPUT_DIM
    _embedding_dim = _EMBEDDING_DIM

    def test_defaults(self) -> None:
        cfg = ResNetConfig()
        assert cfg.num_resnet_layers is None
        assert cfg.output_dim is None
        assert cfg.activation_cls is None
        assert cfg.use_batchnorm is False
        assert cfg.use_layernorm is False
        assert cfg.dropout_p == 0.0
        assert cfg.bias is True

    def test_build_type_and_output_dim(self) -> None:
        cfg = ResNetConfig(num_resnet_layers=2)
        mod = cfg.build(self._input_dim, self._embedding_dim)
        assert isinstance(mod, ResNet)
        assert mod.output_dim == self._input_dim
        assert mod._num_resnet_layers == 2

    def test_build_with_explicit_output_dim(self) -> None:
        cfg = ResNetConfig(num_resnet_layers=2, output_dim=4)
        mod = cfg.build(self._input_dim, self._embedding_dim)
        assert mod.output_dim == 4

    def test_build_forward_roundtrip(self) -> None:
        cfg = ResNetConfig(num_resnet_layers=2)
        mod = cfg.build(self._input_dim, self._embedding_dim).eval()
        x = torch.randn(_BATCH_SIZE, self._input_dim)
        cond = torch.randn(_BATCH_SIZE, self._embedding_dim)
        with torch.no_grad():
            assert mod(x, cond).shape == (_BATCH_SIZE, self._input_dim)

    def test_build_passes_every_field(self) -> None:
        cfg = ResNetConfig(
            num_resnet_layers=2,
            use_batchnorm=True,
            dropout_p=0.25,
            bias=False,
        )
        mod = cfg.build(self._input_dim, self._embedding_dim)
        assert mod._use_batchnorm is True
        assert mod._dropout_p == pytest.approx(0.25)
        assert mod._bias is False

    def test_build_does_not_leak_registry_type(self) -> None:
        """Regression test for the ``self.dict()`` bug shared with the pooling
        configs: the returned dict includes the registry ``type`` key, which
        ``ResNet.__init__`` rejects with ``TypeError``. The fix is
        ``model_dump(exclude={'type'})`` (or ``dict(self)``, matching
        ``MLPConfig.mlp_kwargs``)."""
        cfg = ResNetConfig(num_resnet_layers=2)
        mod = cfg.build(self._input_dim, self._embedding_dim)  # must not raise
        assert isinstance(mod, ResNet)

    def test_build_returns_fresh_instance(self) -> None:
        cfg = ResNetConfig(num_resnet_layers=2)
        a = cfg.build(self._input_dim, self._embedding_dim)
        b = cfg.build(self._input_dim, self._embedding_dim)
        assert a is not b
        for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters(), strict=False):
            assert na == nb
            assert pa is not pb

    def test_build_is_deterministic_in_eval(self) -> None:
        torch.manual_seed(0)
        cfg = ResNetConfig(num_resnet_layers=2, dropout_p=0.5)
        mod = cfg.build(self._input_dim, self._embedding_dim).eval()
        x = torch.randn(_BATCH_SIZE, self._input_dim)
        cond = torch.randn(_BATCH_SIZE, self._embedding_dim)
        with torch.no_grad():
            assert torch.allclose(mod(x, cond), mod(x, cond))
