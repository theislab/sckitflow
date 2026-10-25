"""Tests for the pooling layers and their configs.

Covers :class:`TokenAttentionPooling`, :class:`SeedAttentionPooling`,
:class:`_FunctionalModule` (via the aggregation configs), and the
:class:`BasePoolingLayer` contract they all share.

Coverage axes:

* shape contract (unbatched, batched, extra leading dims, n == 1)
* permutation invariance across the set axis (defining property of pooling)
* determinism / dropout behaviour split across ``train()`` / ``eval()``
* the ``qkv_dim % num_heads`` validation on both attention subclasses
* gradient flow through every learnable
* ``output_dim`` on every built module (required by :class:`SetEncoder`)
* ``BasePoolingLayer`` inheritance for all four pooling types
* mean/sum cross-check (``mean == sum / n``)
* config round-trip: defaults, ``build(input_dim)``, field forwarding

Note on attribute names: the source uses ``_``-prefixed names for the
internal ``Linear`` submodules and the learned query parameter
(``_q``, ``_k``, ``_v``, ``_out``, ``_query``). ``state_dict()`` keys
will be ``_q.weight`` etc. — the tests reflect this.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from sckitflow.core.nn.blocks._pooling import (
    BasePoolingLayer,
    MeanPoolingConfig,
    SeedAttentionPooling,
    SeedAttentionPoolingConfig,
    SumPoolingConfig,
    TokenAttentionPooling,
    TokenAttentionPoolingConfig,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Small but head-divisible default: qkv/v_dim 16 splits cleanly across 2 heads.
_INPUT_DIM = 8
_HEADS = 2
_QKV_DIM = 16

# Shapes exercised by the shape-contract test. ``(*shape[:-2], input_dim)``
# is always the expected output shape for mean/sum/token poolings; seed
# pooling uses ``v_dim`` instead, but the leading dims logic is the same.
_INPUT_SHAPES = [
    pytest.param((5, _INPUT_DIM), id="unbatched"),
    pytest.param((1, _INPUT_DIM), id="unbatched-singleton"),
    pytest.param((4, 5, _INPUT_DIM), id="batched"),
    pytest.param((2, 4, 5, _INPUT_DIM), id="extra-lead-dim"),
]


def _make_token(**overrides: Any) -> TokenAttentionPooling:
    kwargs = {"num_heads": _HEADS, "qkv_dim": _QKV_DIM, **overrides}
    return TokenAttentionPooling(_INPUT_DIM, **kwargs)


def _make_seed(**overrides: Any) -> SeedAttentionPooling:
    # Keep ``v_dim == input_dim`` so tests that overwrite ``_v.weight`` with
    # an identity matrix are shape-correct.
    kwargs = {
        "num_heads": _HEADS,
        "v_dim": _INPUT_DIM,
        "seed_dim": 8,
        **overrides,
    }
    return SeedAttentionPooling(_INPUT_DIM, **kwargs)


# ---------------------------------------------------------------------------
# BasePoolingLayer contract
# ---------------------------------------------------------------------------


class TestBasePoolingLayer:
    """Shared contract: every pooling must be a ``BasePoolingLayer`` and
    expose a positive-int ``output_dim``. :class:`SetEncoder` relies on this
    to size its decoder; a missing implementation used to crash construction.
    """

    def test_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            BasePoolingLayer()  # type: ignore[abstract]

    @pytest.mark.parametrize(
        "factory",
        [
            lambda d: MeanPoolingConfig().build(d),
            lambda d: SumPoolingConfig().build(d),
            lambda d: TokenAttentionPoolingConfig(num_heads=_HEADS, qkv_dim=_QKV_DIM).build(d),
            lambda d: SeedAttentionPoolingConfig(num_heads=_HEADS, v_dim=_INPUT_DIM, seed_dim=8).build(d),
        ],
        ids=["mean", "sum", "token", "seed"],
    )
    def test_all_poolings_inherit_base(self, factory) -> None:
        mod = factory(_INPUT_DIM)
        assert isinstance(mod, BasePoolingLayer)
        assert isinstance(mod, torch.nn.Module)

    @pytest.mark.parametrize(
        "factory",
        [
            lambda d: MeanPoolingConfig().build(d),
            lambda d: SumPoolingConfig().build(d),
            lambda d: TokenAttentionPoolingConfig(num_heads=_HEADS, qkv_dim=_QKV_DIM).build(d),
            lambda d: SeedAttentionPoolingConfig(num_heads=_HEADS, v_dim=_INPUT_DIM, seed_dim=8).build(d),
        ],
        ids=["mean", "sum", "token", "seed"],
    )
    def test_output_dim_is_positive_int(self, factory) -> None:
        mod = factory(_INPUT_DIM)
        assert isinstance(mod.output_dim, int)
        assert mod.output_dim > 0


# ---------------------------------------------------------------------------
# TokenAttentionPooling
# ---------------------------------------------------------------------------


class TestTokenAttentionPooling:
    # ---- construction / structural invariants ----------------------------

    def test_output_dim(self) -> None:
        mod = _make_token()
        assert mod.output_dim == _INPUT_DIM

    def test_registered_children(self) -> None:
        mod = _make_token()
        children = dict(mod.named_children())
        assert {"_q", "_k", "_v", "_out"} <= set(children)
        for name in ("_q", "_k", "_v", "_out"):
            assert isinstance(children[name], torch.nn.Linear), name

    def test_query_parameter_shape_and_registration(self) -> None:
        mod = _make_token()
        params = dict(mod.named_parameters())
        assert "_query" in params
        assert params["_query"].shape == (1, 1, _INPUT_DIM)
        assert params["_query"].requires_grad

    def test_token_flag(self) -> None:
        assert _make_token()._token is True

    # ---- validation -------------------------------------------------------

    @pytest.mark.parametrize("num_heads,qkv_dim", [(3, 16), (5, 16), (7, 16)])
    def test_qkv_dim_must_divide_num_heads(self, num_heads: int, qkv_dim: int) -> None:
        with pytest.raises(ValueError, match="divisible"):
            TokenAttentionPooling(_INPUT_DIM, num_heads=num_heads, qkv_dim=qkv_dim)

    # ---- shape contract ---------------------------------------------------

    @pytest.mark.parametrize("shape", _INPUT_SHAPES)
    def test_forward_shapes(self, shape: tuple[int, ...]) -> None:
        mod = _make_token()
        x = torch.randn(*shape)
        y = mod(x)
        assert y.shape == (*shape[:-2], _INPUT_DIM)

    # ---- set-semantics invariants ----------------------------------------

    @pytest.mark.parametrize("shape", [(4, 5, _INPUT_DIM), (5, _INPUT_DIM)])
    def test_permutation_invariance(self, shape: tuple[int, ...]) -> None:
        """SDPA sums over the key axis, so the pooled output must not depend
        on the order of the set elements. This is the only semantic guarantee
        the layer makes about the set dimension.
        """
        mod = _make_token().eval()
        x = torch.randn(*shape)
        n = shape[-2]
        perm = torch.randperm(n)
        with torch.no_grad():
            y = mod(x)
            y_perm = mod(x[..., perm, :])
        assert torch.allclose(y, y_perm, atol=1e-5)

    def test_query_parameter_influences_output(self) -> None:
        """The token is an active part of the attention; scaling the learned
        query must change the pooled representation. Guards against a wiring
        regression where ``_query`` is created but never fed to ``_q``.
        """
        torch.manual_seed(0)
        mod = _make_token().eval()
        x = torch.randn(4, 5, _INPUT_DIM)
        with torch.no_grad():
            y1 = mod(x)
            mod._query.mul_(3.0)
            y2 = mod(x)
        assert not torch.allclose(y1, y2)

    # ---- dropout ---------------------------------------------------------

    def test_eval_mode_is_deterministic_with_dropout(self) -> None:
        torch.manual_seed(0)
        mod = _make_token(dropout_rate=0.5).eval()
        x = torch.randn(4, 5, _INPUT_DIM)
        with torch.no_grad():
            assert torch.allclose(mod(x), mod(x))

    def test_train_mode_zero_dropout_is_deterministic(self) -> None:
        torch.manual_seed(0)
        mod = _make_token(dropout_rate=0.0).train()
        x = torch.randn(4, 5, _INPUT_DIM)
        with torch.no_grad():
            assert torch.allclose(mod(x), mod(x))

    def test_train_mode_active_dropout_is_stochastic(self) -> None:
        mod = _make_token(dropout_rate=0.5).train()
        x = torch.randn(16, 5, _INPUT_DIM)
        with torch.no_grad():
            # Distinct seeds force distinct dropout masks, so results must differ.
            torch.manual_seed(0)
            y1 = mod(x)
            torch.manual_seed(1)
            y2 = mod(x)
        assert not torch.allclose(y1, y2)

    # ---- gradients -------------------------------------------------------

    def test_all_parameters_receive_gradients(self) -> None:
        mod = _make_token()
        x = torch.randn(4, 5, _INPUT_DIM)
        mod(x).sum().backward()
        for name, p in mod.named_parameters():
            assert p.grad is not None, f"parameter {name!r} received no gradient"


# ---------------------------------------------------------------------------
# SeedAttentionPooling
# ---------------------------------------------------------------------------


class TestSeedAttentionPooling:
    # ---- construction / structural invariants ----------------------------

    def test_output_dim_is_v_dim(self) -> None:
        mod = SeedAttentionPooling(_INPUT_DIM, num_heads=_HEADS, v_dim=32, seed_dim=16)
        assert mod.output_dim == 32

    def test_registered_children(self) -> None:
        mod = _make_seed()
        children = dict(mod.named_children())
        assert {"_q", "_k", "_v"} <= set(children)
        # No output projection: the pooled value IS the output.
        assert "_out" not in children

    def test_query_parameter_shape(self) -> None:
        mod = _make_seed(seed_dim=16)
        params = dict(mod.named_parameters())
        assert params["_query"].shape == (1, 1, 16)

    def test_token_flag(self) -> None:
        assert _make_seed()._token is False

    # ---- validation -------------------------------------------------------

    @pytest.mark.parametrize("num_heads,v_dim", [(3, 16), (5, 16), (7, 16)])
    def test_v_dim_must_divide_num_heads(self, num_heads: int, v_dim: int) -> None:
        with pytest.raises(ValueError, match="divisible"):
            SeedAttentionPooling(_INPUT_DIM, num_heads=num_heads, v_dim=v_dim)

    # ---- shape contract ---------------------------------------------------

    @pytest.mark.parametrize("shape", _INPUT_SHAPES)
    def test_forward_shapes(self, shape: tuple[int, ...]) -> None:
        mod = _make_seed()
        x = torch.randn(*shape)
        y = mod(x)
        assert y.shape == (*shape[:-2], mod.output_dim)

    @pytest.mark.parametrize("shape", [(4, 5, _INPUT_DIM), (5, _INPUT_DIM)])
    def test_permutation_invariance(self, shape: tuple[int, ...]) -> None:
        mod = _make_seed().eval()
        x = torch.randn(*shape)
        n = shape[-2]
        perm = torch.randperm(n)
        with torch.no_grad():
            y = mod(x)
            y_perm = mod(x[..., perm, :])
        assert torch.allclose(y, y_perm, atol=1e-5)

    # ---- the actual math the layer is supposed to do ---------------------

    def _identity_v(self, mod: SeedAttentionPooling) -> None:
        """Freeze ``_v`` to the identity map so ``v(x_i) == x_i`` exactly."""
        with torch.no_grad():
            mod._v.weight.copy_(torch.eye(_INPUT_DIM))
            mod._v.bias.zero_()

    def test_output_is_convex_combination_of_values(self) -> None:
        """``out = sum_i w_i * v(x_i)`` with ``w`` a softmax → the output must
        lie in the coordinate-wise convex hull of ``v(x) = x`` (after the
        identity-``v`` trick). This pins the SDPA reduction, not just its
        shape.
        """
        mod = _make_seed().eval()
        self._identity_v(mod)
        x = torch.randn(4, 10, _INPUT_DIM)
        with torch.no_grad():
            out = mod(x)
        lo = x.min(dim=1).values
        hi = x.max(dim=1).values
        assert torch.all(out >= lo - 1e-5)
        assert torch.all(out <= hi + 1e-5)

    def test_uniform_input_collapses_to_value_regardless_of_cardinality(self) -> None:
        """If every element of the set is the same vector, ``v(x_i)`` is the
        same for all ``i`` and any convex combination is ``v(x_0)``. Also
        checks softmax normalises correctly across variable-length key sets.
        """
        mod = _make_seed().eval()
        self._identity_v(mod)
        v = torch.randn(_INPUT_DIM)
        x3 = v.expand(1, 3, _INPUT_DIM).clone()
        x10 = v.expand(1, 10, _INPUT_DIM).clone()
        with torch.no_grad():
            y3 = mod(x3)
            y10 = mod(x10)
        assert torch.allclose(y3, y10, atol=1e-5)
        assert torch.allclose(y3.squeeze(0), v, atol=1e-5)

    # ---- dropout / gradients ---------------------------------------------

    def test_eval_mode_is_deterministic_with_dropout(self) -> None:
        torch.manual_seed(0)
        mod = _make_seed(dropout_rate=0.5).eval()
        x = torch.randn(4, 5, _INPUT_DIM)
        with torch.no_grad():
            assert torch.allclose(mod(x), mod(x))

    def test_train_mode_active_dropout_is_stochastic(self) -> None:
        mod = _make_seed(dropout_rate=0.5).train()
        x = torch.randn(16, 5, _INPUT_DIM)
        with torch.no_grad():
            torch.manual_seed(0)
            y1 = mod(x)
            torch.manual_seed(1)
            y2 = mod(x)
        assert not torch.allclose(y1, y2)

    def test_all_parameters_receive_gradients(self) -> None:
        mod = _make_seed()
        x = torch.randn(4, 5, _INPUT_DIM)
        mod(x).sum().backward()
        for name, p in mod.named_parameters():
            assert p.grad is not None, f"parameter {name!r} received no gradient"


# ---------------------------------------------------------------------------
# Configs — attention poolings
# ---------------------------------------------------------------------------


class TestTokenAttentionPoolingConfig:
    def test_defaults(self) -> None:
        cfg = TokenAttentionPoolingConfig()
        assert cfg.num_heads == 8
        assert cfg.qkv_dim == 64
        assert cfg.dropout_rate == 0.0

    def test_build_type_and_output_dim(self) -> None:
        cfg = TokenAttentionPoolingConfig(num_heads=_HEADS, qkv_dim=_QKV_DIM)
        mod = cfg.build(_INPUT_DIM)
        assert isinstance(mod, TokenAttentionPooling)
        assert mod.output_dim == _INPUT_DIM
        assert mod._num_heads == _HEADS

    def test_build_forward_roundtrip(self) -> None:
        cfg = TokenAttentionPoolingConfig(num_heads=_HEADS, qkv_dim=_QKV_DIM)
        mod = cfg.build(_INPUT_DIM).eval()
        x = torch.randn(4, 5, _INPUT_DIM)
        with torch.no_grad():
            assert mod(x).shape == (4, _INPUT_DIM)

    def test_build_passes_every_field(self) -> None:
        cfg = TokenAttentionPoolingConfig(num_heads=_HEADS, qkv_dim=_QKV_DIM, dropout_rate=0.25)
        mod = cfg.build(_INPUT_DIM)
        assert mod._num_heads == _HEADS
        assert mod._dropout_rate == pytest.approx(0.25)

    def test_build_returns_fresh_instance(self) -> None:
        cfg = TokenAttentionPoolingConfig(num_heads=_HEADS, qkv_dim=_QKV_DIM)
        a = cfg.build(_INPUT_DIM)
        b = cfg.build(_INPUT_DIM)
        assert a is not b
        assert a._q.weight is not b._q.weight


class TestSeedAttentionPoolingConfig:
    def test_defaults(self) -> None:
        cfg = SeedAttentionPoolingConfig()
        assert cfg.num_heads == 8
        assert cfg.v_dim == 64
        assert cfg.seed_dim == 64
        assert cfg.dropout_rate == 0.0

    def test_build_type_and_output_dim(self) -> None:
        cfg = SeedAttentionPoolingConfig(num_heads=_HEADS, v_dim=32, seed_dim=16)
        mod = cfg.build(_INPUT_DIM)
        assert isinstance(mod, SeedAttentionPooling)
        assert mod.output_dim == 32

    def test_build_forward_roundtrip(self) -> None:
        cfg = SeedAttentionPoolingConfig(num_heads=_HEADS, v_dim=32, seed_dim=16)
        mod = cfg.build(_INPUT_DIM).eval()
        x = torch.randn(4, 5, _INPUT_DIM)
        with torch.no_grad():
            assert mod(x).shape == (4, 32)

    def test_build_passes_every_field(self) -> None:
        cfg = SeedAttentionPoolingConfig(num_heads=_HEADS, v_dim=32, seed_dim=16, dropout_rate=0.25)
        mod = cfg.build(_INPUT_DIM)
        assert mod._num_heads == _HEADS
        assert mod._dropout_rate == pytest.approx(0.25)

    def test_build_returns_fresh_instance(self) -> None:
        cfg = SeedAttentionPoolingConfig(num_heads=_HEADS, v_dim=32, seed_dim=16)
        a = cfg.build(_INPUT_DIM)
        b = cfg.build(_INPUT_DIM)
        assert a is not b
        assert a._k.weight is not b._k.weight


# ---------------------------------------------------------------------------
# MeanPooling / SumPooling (functional aggregations)
# ---------------------------------------------------------------------------


class TestAggregationPoolingConfigs:
    _INPUT_DIM = 8
    _BATCH = 4
    _SET_SIZE = 5

    # ---- shared interface with attention poolings -----------------------

    @pytest.mark.parametrize("cfg_factory", [MeanPoolingConfig, SumPoolingConfig], ids=["mean", "sum"])
    def test_build_returns_module(self, cfg_factory) -> None:
        mod = cfg_factory().build(self._INPUT_DIM)
        assert isinstance(mod, torch.nn.Module)

    @pytest.mark.parametrize("cfg_factory", [MeanPoolingConfig, SumPoolingConfig], ids=["mean", "sum"])
    def test_output_dim_matches_input_dim(self, cfg_factory) -> None:
        mod = cfg_factory().build(self._INPUT_DIM)
        assert mod.output_dim == self._INPUT_DIM

    @pytest.mark.parametrize("cfg_factory", [MeanPoolingConfig, SumPoolingConfig], ids=["mean", "sum"])
    def test_registers_a_child_module(self, cfg_factory) -> None:
        """Regression guard for the ``_net = Identity()`` device: some
        frameworks/optimizers error when a module has neither parameters nor
        children. This asserts at least one child exists.
        """
        mod = cfg_factory().build(self._INPUT_DIM)
        assert len(list(mod.children())) >= 1

    @pytest.mark.parametrize("cfg_factory", [MeanPoolingConfig, SumPoolingConfig], ids=["mean", "sum"])
    def test_has_no_parameters(self, cfg_factory) -> None:
        mod = cfg_factory().build(self._INPUT_DIM)
        assert list(mod.parameters()) == []

    # ---- shape contract --------------------------------------------------

    @pytest.mark.parametrize("cfg_factory", [MeanPoolingConfig, SumPoolingConfig], ids=["mean", "sum"])
    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param((5, 8), id="unbatched"),
            pytest.param((4, 5, 8), id="batched"),
            pytest.param((2, 4, 5, 8), id="extra-lead-dim"),
        ],
    )
    def test_forward_shape(self, cfg_factory, shape) -> None:
        mod = cfg_factory().build(self._INPUT_DIM)
        y = mod(torch.randn(*shape))
        assert y.shape == (*shape[:-2], self._INPUT_DIM)

    # ---- mean semantics --------------------------------------------------

    def test_mean_matches_torch_mean(self) -> None:
        mod = MeanPoolingConfig().build(self._INPUT_DIM)
        x = torch.randn(self._BATCH, self._SET_SIZE, self._INPUT_DIM)
        assert torch.allclose(mod(x), x.mean(dim=-2))

    def test_mean_is_duplication_invariant(self) -> None:
        """``mean(concat(x, x)) == mean(x)`` — the invariant that
        distinguishes mean from sum.
        """
        mod = MeanPoolingConfig().build(self._INPUT_DIM)
        x = torch.randn(self._BATCH, self._SET_SIZE, self._INPUT_DIM)
        x_dup = x.repeat_interleave(2, dim=1)
        assert torch.allclose(mod(x), mod(x_dup), atol=1e-6)

    def test_mean_collapses_identical_elements(self) -> None:
        mod = MeanPoolingConfig().build(self._INPUT_DIM)
        v = torch.randn(self._INPUT_DIM)
        for n in (1, 3, 10):
            x = v.expand(self._BATCH, n, self._INPUT_DIM).clone()
            assert torch.allclose(mod(x), v.expand(self._BATCH, self._INPUT_DIM), atol=1e-6)

    # ---- sum semantics ---------------------------------------------------

    def test_sum_matches_torch_sum(self) -> None:
        mod = SumPoolingConfig().build(self._INPUT_DIM)
        x = torch.randn(self._BATCH, self._SET_SIZE, self._INPUT_DIM)
        assert torch.allclose(mod(x), x.sum(dim=-2))

    def test_sum_scales_linearly_under_duplication(self) -> None:
        """``sum(concat(x, x)) == 2 * sum(x)`` — dual of the mean's
        duplication invariance.
        """
        mod = SumPoolingConfig().build(self._INPUT_DIM)
        x = torch.randn(self._BATCH, self._SET_SIZE, self._INPUT_DIM)
        x_dup = x.repeat_interleave(2, dim=1)
        assert torch.allclose(mod(x_dup), 2 * mod(x), atol=1e-6)

    def test_sum_scales_with_cardinality(self) -> None:
        mod = SumPoolingConfig().build(self._INPUT_DIM)
        v = torch.randn(self._INPUT_DIM)
        for n in (1, 3, 10):
            x = v.expand(self._BATCH, n, self._INPUT_DIM).clone()
            expected = v.expand(self._BATCH, self._INPUT_DIM) * n
            assert torch.allclose(mod(x), expected, atol=1e-6)

    # ---- cross-check -----------------------------------------------------

    def test_mean_equals_sum_over_n(self) -> None:
        """The only difference between the two layers is the divide-by-n."""
        x = torch.randn(self._BATCH, self._SET_SIZE, self._INPUT_DIM)
        mean_mod = MeanPoolingConfig().build(self._INPUT_DIM)
        sum_mod = SumPoolingConfig().build(self._INPUT_DIM)
        assert torch.allclose(mean_mod(x), sum_mod(x) / self._SET_SIZE, atol=1e-6)

    # ---- permutation / dtype / grad -------------------------------------

    @pytest.mark.parametrize("cfg_factory", [MeanPoolingConfig, SumPoolingConfig], ids=["mean", "sum"])
    def test_permutation_invariance(self, cfg_factory) -> None:
        mod = cfg_factory().build(self._INPUT_DIM)
        x = torch.randn(self._BATCH, self._SET_SIZE, self._INPUT_DIM)
        perm = torch.randperm(self._SET_SIZE)
        assert torch.allclose(mod(x), mod(x[:, perm, :]), atol=1e-5)

    @pytest.mark.parametrize("cfg_factory", [MeanPoolingConfig, SumPoolingConfig], ids=["mean", "sum"])
    def test_preserves_dtype(self, cfg_factory) -> None:
        mod = cfg_factory().build(self._INPUT_DIM)
        for dtype in (torch.float32, torch.float64):
            x = torch.randn(self._BATCH, self._SET_SIZE, self._INPUT_DIM, dtype=dtype)
            assert mod(x).dtype == dtype

    @pytest.mark.parametrize("cfg_factory", [MeanPoolingConfig, SumPoolingConfig], ids=["mean", "sum"])
    def test_gradients_flow(self, cfg_factory) -> None:
        mod = cfg_factory().build(self._INPUT_DIM)
        x = torch.randn(self._BATCH, self._SET_SIZE, self._INPUT_DIM, requires_grad=True)
        mod(x).sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# Cross-type: all poolings share the [..., n, D] -> [..., D] contract
# ---------------------------------------------------------------------------


class TestAllPoolingTypes:
    _INPUT_DIM = 8
    _BATCH = 4
    _SET_SIZE = 5

    _FACTORIES = [
        pytest.param(lambda d: MeanPoolingConfig().build(d), id="mean"),
        pytest.param(lambda d: SumPoolingConfig().build(d), id="sum"),
        pytest.param(
            lambda d: TokenAttentionPoolingConfig(num_heads=_HEADS, qkv_dim=_QKV_DIM).build(d),
            id="token",
        ),
        pytest.param(
            lambda d: SeedAttentionPoolingConfig(num_heads=_HEADS, v_dim=_INPUT_DIM, seed_dim=8).build(d),
            id="seed",
        ),
    ]

    @pytest.mark.parametrize("factory", _FACTORIES)
    def test_output_dim_attribute_present(self, factory) -> None:
        """Every pooling config's built module must expose ``output_dim``.
        This is what :class:`SetEncoder` relies on to size its decoder.
        """
        mod = factory(self._INPUT_DIM)
        assert isinstance(mod.output_dim, int)
        assert mod.output_dim > 0

    @pytest.mark.parametrize("factory", _FACTORIES)
    def test_shape_contract(self, factory) -> None:
        mod = factory(self._INPUT_DIM)
        for shape in [
            (self._SET_SIZE, self._INPUT_DIM),
            (self._BATCH, self._SET_SIZE, self._INPUT_DIM),
        ]:
            y = mod(torch.randn(*shape))
            assert y.shape == (*shape[:-2], self._INPUT_DIM)

    @pytest.mark.parametrize("factory", _FACTORIES)
    def test_permutation_invariance(self, factory) -> None:
        """The defining property of any set pooling layer."""
        mod = factory(self._INPUT_DIM).eval()
        x = torch.randn(self._BATCH, self._SET_SIZE, self._INPUT_DIM)
        perm = torch.randperm(self._SET_SIZE)
        with torch.no_grad():
            assert torch.allclose(mod(x), mod(x[:, perm, :]), atol=1e-5)
