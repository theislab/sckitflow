"""Tests for :class:`SetEncoder` and :class:`SetEncoderConfig`.

Covers the full pipeline: config → construction validation → forward shape
matrix → internal wiring → pooling variants → covariate key handling →
``SetEncoderConfig.build`` convenience method.

The pipeline assembled by the encoder is:

    continuous_covariate ──► encoder ──► [..., D_c]
    categorical_covariate ─► encoder ─► projection ─► [..., n, P]
                            └► concatenate (dim=-2) ─► pool ─► [..., P']
    [..., D_c] ⊕ [..., P'] ─► output_layer ─► [..., output_dim]

Where ``P`` is ``pooling_proj_dim`` and ``P'`` is the pooling layer's own
``output_dim`` (which differs from ``P`` for :class:`SeedAttentionPooling`).

Design notes baked into the tests:

* Extra covariates in ``condition_dict`` are **ignored by design** — they
  carry case-control / batch metadata that travels alongside the encoding
  but has no encoder. ``test_extra_covariate_ignored_by_design`` pins this.
* Missing *registered* covariates raise ``ValueError`` per covariate.
* ``projection`` and ``pooling`` modules only exist when categorical
  covariates are present.
* ``pooling_proj_dim=None`` (the default) resolves to the minimum
  ``output_dim`` across the categorical input encoders.
"""

from __future__ import annotations

import pytest
import torch

from sckitflow.core.nn.blocks._mlp import MLPConfig, MLPEncoderConfig
from sckitflow.core.nn.blocks._pooling import (
    MeanPoolingConfig,
    SeedAttentionPoolingConfig,
    SumPoolingConfig,
    TokenAttentionPoolingConfig,
)
from sckitflow.core.nn.composite._set_encoder import SetEncoder, SetEncoderConfig

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

BATCH = 32
N_COMBS = 3
OUTPUT_DIM = 16
POOLING_PROJ_DIM = 5

# (raw_input_dim, encoded_output_dim) per covariate. Distinct encoded dims
# ensure the min() path actually distinguishes them.
COV_DIMS = {
    "condition0": (4, 2),
    "condition1": (8, 4),
    "condition2": (6, 3),
    "condition3": (10, 5),
}


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _dims(
    *,
    continuous=(),
    categorical=(),
    extra_continuous=(),
    extra_categorical=(),
):
    """Per-covariate input dims, as keyword arguments for ``SetEncoder`` / ``SetEncoderConfig.build``.

    ``extra_*`` arguments populate the dims with covariates that have no
    corresponding encoder — a legitimate case (dims can be a superset).
    """
    return {
        "categorical_dims": {k: COV_DIMS[k][0] for k in (*categorical, *extra_categorical)},
        "continuous_dims": {k: COV_DIMS[k][0] for k in (*continuous, *extra_continuous)},
    }


def _encoder_cfg(cov_id: str, *, batchnorm: bool = False) -> MLPEncoderConfig:
    return MLPEncoderConfig(
        mlp=MLPConfig(use_batchnorm=batchnorm),
        output_dim=COV_DIMS[cov_id][1],
    )


def _cfg(
    *,
    continuous=(),
    categorical=(),
    output_dim: int = OUTPUT_DIM,
    pooling=None,
    pooling_proj_dim: int | None = POOLING_PROJ_DIM,
    pooling_proj_bias: bool = True,
    output_layer: MLPConfig | None = None,
    categorical_batchnorm: bool = False,
    continuous_batchnorm: bool = False,
) -> SetEncoderConfig:
    return SetEncoderConfig(
        output_dim=output_dim,
        categorical_input_layers=({k: _encoder_cfg(k, batchnorm=categorical_batchnorm) for k in categorical} or None),
        continuous_input_layers=({k: _encoder_cfg(k, batchnorm=continuous_batchnorm) for k in continuous} or None),
        pooling=pooling,
        pooling_proj_dim=pooling_proj_dim,
        pooling_proj_bias=pooling_proj_bias,
        output_layer=output_layer,
    )


def _inputs(*, continuous=(), categorical=(), batch: int = BATCH):
    """Build a ``condition_dict`` matching the registered covariates."""
    d = {}
    for k in continuous:
        d[k] = torch.randn(batch, COV_DIMS[k][0])
    for k in categorical:
        d[k] = torch.randn(batch, N_COMBS, COV_DIMS[k][0])
    return d


def _build(*, continuous=(), categorical=(), **overrides) -> SetEncoder:
    """Convenience: build encoder and matching dims in one call."""
    cfg = _cfg(continuous=continuous, categorical=categorical, **overrides)
    dims = _dims(continuous=continuous, categorical=categorical)
    return SetEncoder(cfg, **dims)


def _decoder_in_features(encoder: SetEncoder) -> int:
    """Recover the input dimension of the decoder's first Linear."""
    return encoder._set_encoder["output_layer"]._mlp[0].linear.in_features


# ---------------------------------------------------------------------------
# SetEncoderConfig
# ---------------------------------------------------------------------------


class TestSetEncoderConfig:
    def test_requires_output_dim(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SetEncoderConfig()  # type: ignore[call-arg]

    def test_defaults_are_none(self) -> None:
        cfg = SetEncoderConfig(output_dim=OUTPUT_DIM)
        assert cfg.categorical_input_layers is None
        assert cfg.continuous_input_layers is None
        assert cfg.pooling is None
        assert cfg.pooling_proj_dim is None
        assert cfg.output_layer is None

    def test_pooling_proj_bias_default_is_true(self) -> None:
        cfg = SetEncoderConfig(output_dim=OUTPUT_DIM)
        assert cfg.pooling_proj_bias is True

    def test_explicit_values_are_preserved(self) -> None:
        cfg = _cfg(
            continuous=["condition0"],
            categorical=["condition1"],
            output_dim=7,
            pooling=MeanPoolingConfig(),
            pooling_proj_dim=3,
            pooling_proj_bias=False,
        )
        assert cfg.output_dim == 7
        assert set(cfg.categorical_input_layers) == {"condition1"}
        assert set(cfg.continuous_input_layers) == {"condition0"}
        assert isinstance(cfg.pooling, MeanPoolingConfig)
        assert cfg.pooling_proj_dim == 3
        assert cfg.pooling_proj_bias is False


# ---------------------------------------------------------------------------
# SetEncoderConfig.build(...)
# ---------------------------------------------------------------------------


class TestSetEncoderConfigBuild:
    """The ``build`` convenience method must construct a working
    encoder using the config's own fields.
    """

    def test_returns_set_encoder(self) -> None:
        cfg = _cfg(categorical=["condition0"])
        dims = _dims(categorical=["condition0"])
        enc = cfg.build(**dims)
        assert isinstance(enc, SetEncoder)

    def test_uses_config_output_dim(self) -> None:
        cfg = _cfg(categorical=["condition0"], output_dim=7)
        enc = cfg.build(**_dims(categorical=["condition0"])).eval()
        cond = _inputs(categorical=("condition0",))
        with torch.no_grad():
            assert enc(cond).shape == (BATCH, 7)

    def test_uses_config_pooling(self) -> None:
        v_dim = 8
        cfg = _cfg(
            categorical=["condition0"],
            pooling=SeedAttentionPoolingConfig(num_heads=2, v_dim=v_dim, seed_dim=4),
        )
        enc = cfg.build(**_dims(categorical=["condition0"]))
        assert _decoder_in_features(enc) == v_dim

    def test_uses_config_pooling_proj_dim(self) -> None:
        cfg = _cfg(categorical=["condition0"], pooling_proj_dim=7)
        enc = cfg.build(**_dims(categorical=["condition0"]))
        assert enc._set_encoder["projection"]["condition0"].out_features == 7

    def test_equivalent_to_direct_construction(self) -> None:
        """``cfg.build(**dims)`` must produce the same architecture as
        ``SetEncoder(cfg, **dims)`` directly.
        """
        cfg = _cfg(
            continuous=["condition0"],
            categorical=["condition1"],
        )
        dims = _dims(continuous=["condition0"], categorical=["condition1"])
        enc_direct = SetEncoder(cfg, **dims)
        enc_built = cfg.build(**dims)
        # Both produce the same decoder input dim and same module structure
        assert _decoder_in_features(enc_direct) == _decoder_in_features(enc_built)
        assert set(enc_direct._set_encoder.keys()) == set(enc_built._set_encoder.keys())


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestSetEncoderConstruction:
    def test_no_covariates_raises(self) -> None:
        cfg = _cfg()  # neither categorical nor continuous
        with pytest.raises(ValueError, match="Set encoder needs one of"):
            SetEncoder(cfg, **_dims())

    def test_continuous_only(self) -> None:
        """A continuous-only encoder is legitimate; no ``min([])`` crash."""
        enc = _build(continuous=("condition0",))
        assert _decoder_in_features(enc) == COV_DIMS["condition0"][1]

    def test_categorical_only(self) -> None:
        enc = _build(categorical=("condition0",))
        assert _decoder_in_features(enc) == POOLING_PROJ_DIM

    def test_both(self) -> None:
        enc = _build(continuous=("condition0",), categorical=("condition1",))
        expected = COV_DIMS["condition0"][1] + POOLING_PROJ_DIM
        assert _decoder_in_features(enc) == expected

    # ---- dimension validation -------------------------------------------

    def test_missing_categorical_dim_rejected(self) -> None:
        cfg = _cfg(categorical=["condition0"])
        with pytest.raises(ValueError, match="no corresponding data dimensionality"):
            SetEncoder(cfg, **_dims())  # empty dims

    def test_missing_continuous_dim_rejected(self) -> None:
        cfg = _cfg(continuous=["condition0"])
        with pytest.raises(ValueError, match="no corresponding data dimensionality"):
            SetEncoder(cfg, **_dims())

    def test_extra_dim_key_is_ignored(self) -> None:
        """Dims may be a superset of the config — not every covariate needs
        an encoder.
        """
        cfg = _cfg(categorical=["condition0"])
        dims = _dims(categorical=["condition0"], extra_categorical=["condition1"])
        SetEncoder(cfg, **dims)  # must not raise

    # ---- batchnorm guards -----------------------------------------------

    def test_batchnorm_on_categorical_input_rejected(self) -> None:
        """Categorical covariates are one row per batch, so batchnorm is
        meaningless. The guard reads ``cfg.mlp.use_batchnorm`` on the inner
        ``MLPConfig`` (regression: it must not read ``cfg.use_batchnorm``
        directly on the outer ``MLPEncoderConfig``).
        """
        cfg = _cfg(categorical=["condition0"], categorical_batchnorm=True)
        with pytest.raises(ValueError, match="only ever see one row per batch"):
            SetEncoder(cfg, **_dims(categorical=["condition0"]))

    def test_batchnorm_on_output_rejected_when_no_continuous(self) -> None:
        """Nothing per-observation reaches the output layer, so batchnorm is
        meaningless there too.
        """
        cfg = _cfg(
            categorical=["condition0"],
            output_layer=MLPConfig(use_batchnorm=True),
        )
        with pytest.raises(ValueError, match="only ever see one row per batch"):
            SetEncoder(cfg, **_dims(categorical=["condition0"]))

    def test_batchnorm_on_output_allowed_when_continuous_present(self) -> None:
        """With continuous covariates, the output layer sees per-observation
        rows, so batchnorm is valid.
        """
        cfg = _cfg(
            continuous=["condition0"],
            categorical=["condition1"],
            output_layer=MLPConfig(use_batchnorm=True),
        )
        SetEncoder(
            cfg,
            **_dims(continuous=["condition0"], categorical=["condition1"]),
        )

    def test_batchnorm_on_continuous_input_allowed(self) -> None:
        cfg = _cfg(continuous=["condition0"], continuous_batchnorm=True)
        SetEncoder(cfg, **_dims(continuous=["condition0"]))

    def test_multiple_categorical_batchnorm_names_all_flagged(self) -> None:
        """The error message should list every offending covariate."""
        cfg = _cfg(
            categorical=["condition0", "condition1"],
            categorical_batchnorm=True,
        )
        with pytest.raises(ValueError, match="condition0.*condition1|condition1.*condition0"):
            SetEncoder(
                cfg,
                **_dims(categorical=["condition0", "condition1"]),
            )


# ---------------------------------------------------------------------------
# Forward shapes
# ---------------------------------------------------------------------------


class TestSetEncoderForwardShapes:
    @pytest.mark.parametrize(
        "continuous,categorical",
        [
            ((), ("condition0",)),
            ((), ("condition0", "condition1")),
            (("condition0",), ()),
            (("condition0", "condition1"), ()),
            (("condition0",), ("condition1",)),
            (("condition0", "condition1"), ("condition2",)),
        ],
        ids=[
            "cat-1",
            "cat-2",
            "cont-1",
            "cont-2",
            "cont1-cat1",
            "cont2-cat1",
        ],
    )
    def test_forward_shape(self, continuous, categorical) -> None:
        enc = _build(continuous=continuous, categorical=categorical).eval()
        cond = _inputs(continuous=continuous, categorical=categorical)
        with torch.no_grad():
            out = enc(cond)
        assert out.shape == (BATCH, OUTPUT_DIM)

    def test_leading_batch_axes(self) -> None:
        """Arbitrary leading dims must be preserved: ``[N, B, ...]`` works."""
        enc = _build(
            continuous=("condition0",),
            categorical=("condition1",),
        ).eval()
        for lead in [(BATCH,), (2, BATCH)]:
            cond = {
                "condition0": torch.randn(*lead, COV_DIMS["condition0"][0]),
                "condition1": torch.randn(*lead, N_COMBS, COV_DIMS["condition1"][0]),
            }
            with torch.no_grad():
                assert enc(cond).shape == (*lead, OUTPUT_DIM)

    def test_group_row_broadcasts_against_per_observation(self) -> None:
        """A group's categorical covariate is one row ``(1, n, d)``; continuous
        ones are one row per observation. Broadcasting must give the same
        output as repeating the group row.
        """
        enc = _build(
            continuous=("condition0",),
            categorical=("condition1",),
        ).eval()
        cond = {
            "condition0": torch.randn(BATCH, COV_DIMS["condition0"][0]),
            "condition1": torch.randn(1, N_COMBS, COV_DIMS["condition1"][0]),
        }
        repeated = {**cond, "condition1": cond["condition1"].expand(BATCH, -1, -1)}
        with torch.no_grad():
            torch.testing.assert_close(enc(cond), enc(repeated))

    def test_custom_output_dim(self) -> None:
        enc = _build(categorical=("condition0",), output_dim=7).eval()
        cond = _inputs(categorical=("condition0",))
        with torch.no_grad():
            assert enc(cond).shape == (BATCH, 7)


# ---------------------------------------------------------------------------
# Internal wiring
# ---------------------------------------------------------------------------


class TestSetEncoderInternals:
    def test_module_dict_structure_categorical(self) -> None:
        enc = _build(categorical=("condition0",))
        assert set(enc._set_encoder.keys()) == {
            "input_layers",
            "projection",
            "pooling",
            "output_layer",
        }
        assert set(enc._set_encoder["input_layers"].keys()) == {
            "continuous",
            "categorical",
        }

    def test_module_dict_structure_continuous_only(self) -> None:
        """Without categorical covariates there is nothing to project or pool;
        the corresponding modules must be absent, not dummy placeholders.
        """
        enc = _build(continuous=("condition0",))
        assert set(enc._set_encoder.keys()) == {"input_layers", "output_layer"}

    def test_projection_layers_only_for_categorical(self) -> None:
        enc = _build(
            continuous=("condition0",),
            categorical=("condition1",),
        )
        proj = enc._set_encoder["projection"]
        assert set(proj.keys()) == {"condition1"}
        assert isinstance(proj["condition1"], torch.nn.Linear)
        assert proj["condition1"].in_features == COV_DIMS["condition1"][1]
        assert proj["condition1"].out_features == POOLING_PROJ_DIM

    def test_projection_bias_propagated(self) -> None:
        enc = _build(categorical=("condition0",), pooling_proj_bias=False)
        assert enc._set_encoder["projection"]["condition0"].bias is None

        enc = _build(categorical=("condition0",), pooling_proj_bias=True)
        assert enc._set_encoder["projection"]["condition0"].bias is not None

    def test_input_encoders_nested_by_type(self) -> None:
        enc = _build(
            continuous=("condition0",),
            categorical=("condition1",),
        )
        input_layers = enc._set_encoder["input_layers"]
        assert set(input_layers["continuous"].keys()) == {"condition0"}
        assert set(input_layers["categorical"].keys()) == {"condition1"}

    # ---- decoder input dim ---------------------------------------------

    @pytest.mark.parametrize(
        "continuous,categorical,expected",
        [
            ((), ("condition0",), POOLING_PROJ_DIM),
            (("condition0",), (), COV_DIMS["condition0"][1]),
            (
                ("condition0",),
                ("condition1",),
                POOLING_PROJ_DIM + COV_DIMS["condition0"][1],
            ),
            (
                ("condition0", "condition1"),
                (),
                COV_DIMS["condition0"][1] + COV_DIMS["condition1"][1],
            ),
            (
                ("condition0", "condition1"),
                ("condition2",),
                POOLING_PROJ_DIM + COV_DIMS["condition0"][1] + COV_DIMS["condition1"][1],
            ),
        ],
        ids=[
            "cat-only",
            "cont-only",
            "cont1-cat1",
            "cont2",
            "cont2-cat1",
        ],
    )
    def test_decoder_input_dim(self, continuous, categorical, expected) -> None:
        enc = _build(continuous=continuous, categorical=categorical)
        assert _decoder_in_features(enc) == expected


# ---------------------------------------------------------------------------
# Pooling projection dim resolution
# ---------------------------------------------------------------------------


class TestPoolingProjectionDim:
    def test_explicit_dim_used(self) -> None:
        enc = _build(categorical=("condition0",), pooling_proj_dim=7)
        assert enc._set_encoder["projection"]["condition0"].out_features == 7
        assert _decoder_in_features(enc) == 7

    def test_default_dim_is_min_of_categorical_encoders(self) -> None:
        """``pooling_proj_dim=None`` → ``min`` of the categorical encoders'
        ``output_dim``. Regression: ``min([c.output_dim for c in dict])``
        iterates *keys* (strings) and raises ``AttributeError``; the fix
        is ``for c in dict.values()``.
        """
        enc = _build(
            categorical=("condition0", "condition1"),
            pooling_proj_dim=None,
        )
        expected = min(
            COV_DIMS["condition0"][1],
            COV_DIMS["condition1"][1],
        )
        for cov in ("condition0", "condition1"):
            assert enc._set_encoder["projection"][cov].out_features == expected
        assert _decoder_in_features(enc) == expected

    def test_default_dim_with_three_categorical_covariates(self) -> None:
        """Also check the min path with a 3-way min."""
        enc = _build(
            categorical=("condition0", "condition1", "condition2"),
            pooling_proj_dim=None,
        )
        expected = min(
            COV_DIMS["condition0"][1],
            COV_DIMS["condition1"][1],
            COV_DIMS["condition2"][1],
        )
        assert enc._set_encoder["projection"]["condition0"].out_features == expected


# ---------------------------------------------------------------------------
# Pooling variant output dims
# ---------------------------------------------------------------------------

_POOLINGS = [
    pytest.param(None, POOLING_PROJ_DIM, id="mean-default"),
    pytest.param(MeanPoolingConfig(), POOLING_PROJ_DIM, id="mean"),
    pytest.param(SumPoolingConfig(), POOLING_PROJ_DIM, id="sum"),
    pytest.param(
        TokenAttentionPoolingConfig(num_heads=2, qkv_dim=8),
        POOLING_PROJ_DIM,
        id="token",
    ),
    pytest.param(
        SeedAttentionPoolingConfig(num_heads=2, v_dim=8, seed_dim=4),
        8,
        id="seed",
    ),
]


class TestPoolingVariants:
    @pytest.mark.parametrize("pooling,expected_dim", _POOLINGS)
    def test_pooling_layer_exposes_correct_output_dim(self, pooling, expected_dim: int) -> None:
        """Every pooling config's built module must expose ``output_dim``.
        The set encoder relies on this to size the decoder — a missing
        ``output_dim`` on ``_FunctionalModule`` used to crash construction.
        """
        enc = _build(categorical=("condition0",), pooling=pooling)
        pooling_layer = enc._set_encoder["pooling"]
        assert hasattr(pooling_layer, "output_dim")
        assert pooling_layer.output_dim == expected_dim

    @pytest.mark.parametrize("pooling,expected_dim", _POOLINGS)
    def test_decoder_sized_from_pooling_output_dim(self, pooling, expected_dim: int) -> None:
        enc = _build(categorical=("condition0",), pooling=pooling)
        assert _decoder_in_features(enc) == expected_dim

    @pytest.mark.parametrize("pooling,expected_dim", _POOLINGS)
    def test_forward_runs(self, pooling, expected_dim: int) -> None:
        enc = _build(categorical=("condition0",), pooling=pooling).eval()
        cond = _inputs(categorical=("condition0",))
        with torch.no_grad():
            assert enc(cond).shape == (BATCH, OUTPUT_DIM)

    def test_seed_pooling_with_continuous_dims(self) -> None:
        """With ``v_dim=8`` and a continuous covariate of dim 2, the decoder
        input must be ``8 + 2 = 10``, not ``POOLING_PROJ_DIM + 2``.
        """
        v_dim = 8
        enc = _build(
            continuous=("condition0",),
            categorical=("condition1",),
            pooling=SeedAttentionPoolingConfig(num_heads=2, v_dim=v_dim, seed_dim=4),
        )
        assert _decoder_in_features(enc) == v_dim + COV_DIMS["condition0"][1]

    def test_attention_uneven_heads_rejected(self) -> None:
        cfg = _cfg(
            categorical=["condition0"],
            pooling=TokenAttentionPoolingConfig(num_heads=3, qkv_dim=8),
        )
        with pytest.raises(ValueError, match="divisible"):
            SetEncoder(cfg, **_dims(categorical=["condition0"]))


# ---------------------------------------------------------------------------
# Permutation invariance
# ---------------------------------------------------------------------------


class TestSetEncoderPermutationInvariance:
    @pytest.mark.parametrize(
        "pooling,_",
        [
            (None, 0),
            (MeanPoolingConfig(), 0),
            (SumPoolingConfig(), 0),
            (TokenAttentionPoolingConfig(num_heads=2, qkv_dim=8), 0),
            (SeedAttentionPoolingConfig(num_heads=2, v_dim=8, seed_dim=4), 0),
        ],
        ids=["mean-default", "mean", "sum", "token", "seed"],
    )
    def test_categorical_set_order_independent(self, pooling, _) -> None:
        """The pooled representation must not depend on the order of the
        set elements. This is the defining property of set pooling.
        """
        enc = _build(
            continuous=("condition1",),
            categorical=("condition0",),
            pooling=pooling,
        ).eval()
        cond = _inputs(continuous=("condition1",), categorical=("condition0",))
        flipped = {**cond, "condition0": cond["condition0"].flip(-2)}
        with torch.no_grad():
            torch.testing.assert_close(enc(cond), enc(flipped), atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Attention pooling specifics
# ---------------------------------------------------------------------------


class TestSetEncoderAttentionPooling:
    def test_seed_pooling_query_trains(self) -> None:
        """Backprop through the encoder must reach the seed query parameter.
        Also validates the ``_query`` (underscore-prefixed) attribute name.
        """
        enc = _build(
            continuous=("condition1",),
            categorical=("condition0",),
            pooling=SeedAttentionPoolingConfig(num_heads=2, v_dim=8, seed_dim=4),
        )
        cond = _inputs(continuous=("condition1",), categorical=("condition0",))
        enc(cond).sum().backward()
        assert enc._set_encoder["pooling"]._query.grad is not None

    def test_token_pooling_query_trains(self) -> None:
        enc = _build(
            categorical=("condition0",),
            pooling=TokenAttentionPoolingConfig(num_heads=2, qkv_dim=8),
        )
        cond = _inputs(categorical=("condition0",))
        enc(cond).sum().backward()
        assert enc._set_encoder["pooling"]._query.grad is not None


# ---------------------------------------------------------------------------
# Covariate key validation
# ---------------------------------------------------------------------------


class TestSetEncoderCovariateValidation:
    def test_missing_categorical_covariate_raises(self) -> None:
        enc = _build(
            continuous=("condition0",),
            categorical=("condition1",),
        )
        with pytest.raises(ValueError, match="Categorical covariate condition1 not found"):
            enc({"condition0": torch.randn(BATCH, COV_DIMS["condition0"][0])})

    def test_missing_continuous_covariate_raises(self) -> None:
        enc = _build(
            continuous=("condition0",),
            categorical=("condition1",),
        )
        with pytest.raises(ValueError, match="Continuous covariate condition0 not found"):
            enc({"condition1": torch.randn(BATCH, N_COMBS, COV_DIMS["condition1"][0])})

    def test_extra_covariate_ignored_by_design(self) -> None:
        """Extra keys in ``condition_dict`` that have no encoder are silently
        ignored. Supports case-control workflows where grouping metadata
        travels through ``condition_dict`` without being encoded.
        """
        enc = _build(continuous=("condition0",)).eval()
        cond_with_extra = {
            "condition0": torch.randn(BATCH, COV_DIMS["condition0"][0]),
            "case_id": torch.randn(BATCH, 1),  # not registered, ignored
        }
        cond_clean = {"condition0": cond_with_extra["condition0"]}
        with torch.no_grad():
            out_extra = enc(cond_with_extra)
            out_clean = enc(cond_clean)
        assert out_extra.shape == (BATCH, OUTPUT_DIM)
        torch.testing.assert_close(out_extra, out_clean)

    def test_empty_condition_dict_raises(self) -> None:
        enc = _build(categorical=("condition0",))
        with pytest.raises(ValueError, match="Categorical covariate condition0 not found"):
            enc({})

    def test_missing_multiple_covariates_reports_first(self) -> None:
        """Missing covariate errors mention a specific covariate name."""
        enc = _build(
            continuous=("condition0",),
            categorical=("condition1",),
        )
        with pytest.raises(ValueError, match="condition0"):
            enc({"condition1": torch.randn(BATCH, N_COMBS, COV_DIMS["condition1"][0])})


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


class TestSetEncoderGradients:
    def test_all_parameters_receive_gradients(self) -> None:
        """Every parameter (input encoders, projections, pooling, output
        decoder) must receive gradient on a full forward+backward.
        """
        enc = _build(
            continuous=("condition0",),
            categorical=("condition1",),
        )
        cond = _inputs(continuous=("condition0",), categorical=("condition1",))
        enc(cond).sum().backward()
        for name, p in enc.named_parameters():
            assert p.grad is not None, f"parameter {name!r} received no gradient"


# ---------------------------------------------------------------------------
# Registry / config integration
# ---------------------------------------------------------------------------


class TestSetEncoderConfigRegistry:
    def test_is_a_component(self) -> None:
        from scfit.registry import Component

        assert issubclass(SetEncoderConfig, Component)

    def test_config_with_pooling_round_trips(self) -> None:
        """A nested ``PoolingLayerConfig`` must survive model_dump round-trip."""
        cfg = _cfg(
            categorical=["condition0"],
            pooling=SeedAttentionPoolingConfig(num_heads=2, v_dim=8, seed_dim=4),
        )
        payload = cfg.model_dump()
        rebuilt = SetEncoderConfig.model_validate(payload)
        assert isinstance(rebuilt.pooling, SeedAttentionPoolingConfig)
        assert rebuilt.pooling.num_heads == 2
        assert rebuilt.pooling.v_dim == 8

    def test_config_without_pooling_round_trips(self) -> None:
        cfg = _cfg(categorical=["condition0"])
        payload = cfg.model_dump()
        rebuilt = SetEncoderConfig.model_validate(payload)
        assert rebuilt.pooling is None

    def test_config_with_nested_encoder_round_trips(self) -> None:
        cfg = _cfg(
            continuous=["condition0"],
            categorical=["condition1"],
        )
        payload = cfg.model_dump()
        rebuilt = SetEncoderConfig.model_validate(payload)
        assert isinstance(rebuilt.categorical_input_layers["condition1"], MLPEncoderConfig)
        assert rebuilt.categorical_input_layers["condition1"].output_dim == COV_DIMS["condition1"][1]
        assert rebuilt.continuous_input_layers["condition0"].output_dim == COV_DIMS["condition0"][1]

    def test_rebuilt_config_builds_same_architecture(self) -> None:
        """A round-tripped config must produce the same architecture."""
        cfg = _cfg(
            continuous=["condition0"],
            categorical=["condition1"],
            pooling=MeanPoolingConfig(),
            pooling_proj_dim=7,
        )
        rebuilt = SetEncoderConfig.model_validate(cfg.model_dump())
        dims = _dims(continuous=["condition0"], categorical=["condition1"])
        enc_orig = cfg.build(**dims)
        enc_rebuilt = rebuilt.build(**dims)
        assert _decoder_in_features(enc_orig) == _decoder_in_features(enc_rebuilt)


# ---------------------------------------------------------------------------
# Combination of categorical and continuous covariates
# ---------------------------------------------------------------------------
class TestSetEncoderCombination:
    """Tests specifically exercising the categorical + continuous pipeline.

    The forward contract is::

        encoder_input = concat([pooled_categorical, concatenated_continuous], dim=-1)
        encoder_output = output_layer(encoder_input)

    where::

        pooled_categorical   = pooling(concat([proj_i(cat_encoder_i(x_i)) for i], dim=-2))
        concatenated_continuous = concat([cont_encoder_j(x_j) for j], dim=-1)

    So the concatenation order is **pooled first, continuous second**, and
    continuous covariates *bypass* the projection layer. These are the two
    invariants this class tests.
    """

    # ---- helper ----------------------------------------------------------

    @staticmethod
    def _decoder_input(encoder: SetEncoder, condition_dict) -> torch.Tensor:
        """Capture the tensor fed to the output layer via a forward pre-hook."""
        captured: dict[str, torch.Tensor] = {}

        def hook(module, args):
            captured["input"] = args[0].detach()

        handle = encoder._set_encoder["output_layer"].register_forward_pre_hook(hook)
        try:
            with torch.no_grad():
                encoder(condition_dict)
        finally:
            handle.remove()
        return captured["input"]

    # ---- concatenation order ---------------------------------------------

    def test_decoder_input_is_pooled_then_continuous(self) -> None:
        """The decoder input's columns are ``[pooled_categorical, continuous]``
        in that exact order, with no interleaving.
        """
        torch.manual_seed(0)
        enc = _build(
            continuous=("condition0",),
            categorical=("condition1",),
            pooling=MeanPoolingConfig(),
        ).eval()
        cond = _inputs(continuous=("condition0",), categorical=("condition1",))
        decoder_in = self._decoder_input(enc, cond)

        with torch.no_grad():
            cat_enc = enc._set_encoder["input_layers"]["categorical"]["condition1"](cond["condition1"])
            cat_proj = enc._set_encoder["projection"]["condition1"](cat_enc)
            pooled = enc._set_encoder["pooling"](cat_proj)  # (B, P)
            cont = enc._set_encoder["input_layers"]["continuous"]["condition0"](cond["condition0"])  # (B, D_c)

        P = pooled.shape[-1]
        D_c = cont.shape[-1]
        assert decoder_in.shape == (BATCH, P + D_c)
        # The first P columns must be the pooled categorical representation
        torch.testing.assert_close(decoder_in[:, :P], pooled)
        # The remaining columns must be the continuous representation verbatim
        torch.testing.assert_close(decoder_in[:, P:], cont)

    def test_multiple_continuous_concatenate_in_registration_order(self) -> None:
        """Two continuous covariates concatenate in the order they appear
        in ``continuous_input_layers`` (dict insertion order).
        """
        torch.manual_seed(0)
        enc = _build(continuous=("condition0", "condition1")).eval()
        cond = _inputs(continuous=("condition0", "condition1"))
        decoder_in = self._decoder_input(enc, cond)

        with torch.no_grad():
            cont0 = enc._set_encoder["input_layers"]["continuous"]["condition0"](cond["condition0"])
            cont1 = enc._set_encoder["input_layers"]["continuous"]["condition1"](cond["condition1"])

        D0 = cont0.shape[-1]
        D1 = cont1.shape[-1]
        assert decoder_in.shape == (BATCH, D0 + D1)
        torch.testing.assert_close(decoder_in[:, :D0], cont0)
        torch.testing.assert_close(decoder_in[:, D0:], cont1)

    # ---- projection behaviour --------------------------------------------

    def test_continuous_skips_projection_layer(self) -> None:
        """Continuous covariates are **not** projected — only categorical ones
        are. Verified structurally (no projection entry) and semantically
        (the encoder output appears verbatim in the decoder input).
        """
        enc = _build(
            continuous=("condition0",),
            categorical=("condition1",),
            pooling=MeanPoolingConfig(),
        ).eval()

        # Structural: no projection layer for the continuous covariate
        proj_layer = enc._set_encoder["projection"]
        assert "condition1" in proj_layer
        assert "condition0" not in proj_layer

        # Semantic: continuous encoder output appears verbatim
        cond = _inputs(continuous=("condition0",), categorical=("condition1",))
        decoder_in = self._decoder_input(enc, cond)
        with torch.no_grad():
            cont = enc._set_encoder["input_layers"]["continuous"]["condition0"](cond["condition0"])
        torch.testing.assert_close(decoder_in[:, -cont.shape[-1] :], cont)

    def test_categorical_goes_through_projection(self) -> None:
        """Categorical covariates **do** go through the projection — the raw
        encoder output is not visible in the decoder input.
        """
        torch.manual_seed(0)
        enc = _build(
            categorical=("condition1",),
            pooling=MeanPoolingConfig(),
            pooling_proj_dim=POOLING_PROJ_DIM,
        ).eval()
        # Encoded dim is 4 (from COV_DIMS), projected dim is POOLING_PROJ_DIM (5)
        assert COV_DIMS["condition1"][1] != POOLING_PROJ_DIM

        cond = _inputs(categorical=("condition1",))
        decoder_in = self._decoder_input(enc, cond)
        assert decoder_in.shape == (BATCH, POOLING_PROJ_DIM)
        # The decoder input is not the mean of the raw encoder outputs
        with torch.no_grad():
            raw = enc._set_encoder["input_layers"]["categorical"]["condition1"](cond["condition1"])
            raw_mean = raw.mean(dim=-2)
        assert raw_mean.shape[-1] == COV_DIMS["condition1"][1]
        assert decoder_in.shape[-1] == POOLING_PROJ_DIM
        # They can't be equal — different dims
        assert decoder_in.shape[-1] != raw_mean.shape[-1]

    # ---- zeroing isolation -----------------------------------------------

    @pytest.mark.parametrize(
        "pooling",
        [MeanPoolingConfig(), SumPoolingConfig()],
        ids=["mean", "sum"],
    )
    def test_zeroing_categorical_zeroes_only_pooled_portion(self, pooling) -> None:
        """Zeroing the categorical encoder + projection bias zeros the pooled
        portion of the decoder input while leaving the continuous portion
        untouched. Limited to linear poolings (mean/sum); attention poolings
        have a non-zero bias contribution at zero input.
        """
        torch.manual_seed(0)
        enc = _build(
            continuous=("condition0",),
            categorical=("condition1",),
            pooling=pooling,
        ).eval()

        with torch.no_grad():
            for p in enc._set_encoder["input_layers"]["categorical"]["condition1"].parameters():
                p.zero_()
            # Zero projection bias so zeroed input stays zeroed
            enc._set_encoder["projection"]["condition1"].bias.zero_()

        cond = _inputs(continuous=("condition0",), categorical=("condition1",))
        decoder_in = self._decoder_input(enc, cond)
        P = enc._set_encoder["pooling"].output_dim
        # Pooled portion: zero
        torch.testing.assert_close(decoder_in[:, :P], torch.zeros(BATCH, P, dtype=decoder_in.dtype))
        # Continuous portion: non-zero (the encoder output is generic)
        assert not torch.allclose(decoder_in[:, P:], torch.zeros_like(decoder_in[:, P:]))

    def test_zeroing_continuous_zeroes_only_continuous_portion(self) -> None:
        """Zeroing the continuous encoder zeros only the trailing columns of
        the decoder input while leaving the pooled portion untouched.
        """
        torch.manual_seed(0)
        enc = _build(
            continuous=("condition0",),
            categorical=("condition1",),
            pooling=MeanPoolingConfig(),
        ).eval()

        with torch.no_grad():
            for p in enc._set_encoder["input_layers"]["continuous"]["condition0"].parameters():
                p.zero_()

        cond = _inputs(continuous=("condition0",), categorical=("condition1",))
        decoder_in = self._decoder_input(enc, cond)
        P = enc._set_encoder["pooling"].output_dim
        D_c = COV_DIMS["condition0"][1]
        # Continuous portion: zero
        torch.testing.assert_close(decoder_in[:, -D_c:], torch.zeros(BATCH, D_c, dtype=decoder_in.dtype))
        # Pooled portion: non-zero
        assert not torch.allclose(decoder_in[:, :P], torch.zeros_like(decoder_in[:, :P]))

    # ---- full composition verification -----------------------------------

    def test_full_composition_manual(self) -> None:
        """Recompute every step manually and verify the result equals the
        encoder output. This is the strongest single test: it exercises
        encoder → projection → pooling → concatenation → output layer in
        one assertion.
        """
        torch.manual_seed(0)
        enc = _build(
            continuous=("condition0",),
            categorical=("condition1",),
            pooling=MeanPoolingConfig(),
        ).eval()
        cond = _inputs(continuous=("condition0",), categorical=("condition1",))

        with torch.no_grad():
            # Manual forward
            cat_enc = enc._set_encoder["input_layers"]["categorical"]["condition1"](cond["condition1"])
            cat_proj = enc._set_encoder["projection"]["condition1"](cat_enc)
            pooled = enc._set_encoder["pooling"](cat_proj)
            cont_enc = enc._set_encoder["input_layers"]["continuous"]["condition0"](cond["condition0"])
            decoder_in = torch.cat([pooled, cont_enc], dim=-1)
            expected = enc._set_encoder["output_layer"](decoder_in)

            actual = enc(cond)

        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)

    # ---- sensitivity -----------------------------------------------------

    def test_output_depends_on_each_component(self) -> None:
        """Changing only the continuous or only the categorical input changes
        the output — confirms both contributions reach the decoder.
        """
        torch.manual_seed(0)
        enc = _build(
            continuous=("condition0",),
            categorical=("condition1",),
            pooling=MeanPoolingConfig(),
        ).eval()
        cond_a = _inputs(continuous=("condition0",), categorical=("condition1",))
        cond_b = {**cond_a, "condition0": torch.randn_like(cond_a["condition0"])}
        cond_c = {**cond_a, "condition1": torch.randn_like(cond_a["condition1"])}

        with torch.no_grad():
            y_a = enc(cond_a)
            assert not torch.allclose(y_a, enc(cond_b))
            assert not torch.allclose(y_a, enc(cond_c))

    # ---- broadcasting ----------------------------------------------------

    def test_categorical_singleton_batch_broadcasts(self) -> None:
        """Categorical ``(1, n, d)`` broadcasts against continuous ``(B, d)``."""
        torch.manual_seed(0)
        enc = _build(
            continuous=("condition0",),
            categorical=("condition1",),
            pooling=MeanPoolingConfig(),
        ).eval()
        cond = {
            "condition0": torch.randn(BATCH, COV_DIMS["condition0"][0]),
            "condition1": torch.randn(1, N_COMBS, COV_DIMS["condition1"][0]),
        }
        repeated = {**cond, "condition1": cond["condition1"].expand(BATCH, -1, -1)}
        with torch.no_grad():
            torch.testing.assert_close(enc(cond), enc(repeated))

    def test_mixed_lead_dims_are_rejected(self) -> None:
        """Asymmetric lead dims (extra on continuous only, or extra on
        categorical only) are not supported: ``make_concatenation_possible``
        right-aligns shapes and cannot left-pad a lower-rank source.

        The supported case is "both sides have the same lead dims" — see
        ``TestSetEncoderForwardShapes::test_leading_batch_axes``. This test
        documents the current boundary so a future API change is visible.
        """
        enc = _build(
            continuous=("condition0",),
            categorical=("condition1",),
            pooling=MeanPoolingConfig(),
        ).eval()
        cond = {
            "condition0": torch.randn(2, BATCH, COV_DIMS["condition0"][0]),
            "condition1": torch.randn(BATCH, N_COMBS, COV_DIMS["condition1"][0]),
        }
        with pytest.raises((ValueError, RuntimeError), match="shape|dimension"):
            enc(cond)

    def test_extra_lead_dim_on_both_sides_broadcasts(self) -> None:
        """Both covariates carrying the same lead dim ``(2, B, ...)`` works.
        The complementary positive case; the symmetric lead-dims path is
        already exercised by ``test_leading_batch_axes`` but called out here
        next to the rejection test so the pair is obvious.
        """
        enc = _build(
            continuous=("condition0",),
            categorical=("condition1",),
            pooling=MeanPoolingConfig(),
        ).eval()
        cond = {
            "condition0": torch.randn(2, BATCH, COV_DIMS["condition0"][0]),
            "condition1": torch.randn(2, BATCH, N_COMBS, COV_DIMS["condition1"][0]),
        }
        with torch.no_grad():
            assert enc(cond).shape == (2, BATCH, OUTPUT_DIM)

    # ---- decoder sizing formula ------------------------------------------

    @pytest.mark.parametrize(
        "categorical,continuous",
        [
            (("condition0",), ("condition1",)),
            (("condition0", "condition1"), ("condition2",)),
            (("condition2",), ("condition0", "condition1")),
            (("condition0", "condition1"), ("condition2", "condition3")),
        ],
        ids=["cat1-cont1", "cat2-cont1", "cat1-cont2", "cat2-cont2"],
    )
    @pytest.mark.parametrize(
        "pooling,pooled_dim",
        [
            (None, POOLING_PROJ_DIM),
            (MeanPoolingConfig(), POOLING_PROJ_DIM),
            (SumPoolingConfig(), POOLING_PROJ_DIM),
            (TokenAttentionPoolingConfig(num_heads=2, qkv_dim=8), POOLING_PROJ_DIM),
            (SeedAttentionPoolingConfig(num_heads=2, v_dim=8, seed_dim=4), 8),
        ],
        ids=["mean-default", "mean", "sum", "token", "seed"],
    )
    def test_decoder_input_dim_matches_formula(self, categorical, continuous, pooling, pooled_dim) -> None:
        """``decoder_input_dim = pooling.output_dim + sum(continuous encoder outputs)``.
        Checked across 4 covariate-count combinations × 5 pooling variants.
        """
        enc = _build(
            continuous=continuous,
            categorical=categorical,
            pooling=pooling,
        )
        expected = pooled_dim + sum(COV_DIMS[k][1] for k in continuous)
        assert _decoder_in_features(enc) == expected

    # ---- all pooling types ----------------------------------------------

    @pytest.mark.parametrize(
        "pooling",
        [
            None,
            MeanPoolingConfig(),
            SumPoolingConfig(),
            TokenAttentionPoolingConfig(num_heads=2, qkv_dim=8),
            SeedAttentionPoolingConfig(num_heads=2, v_dim=8, seed_dim=4),
        ],
        ids=["mean-default", "mean", "sum", "token", "seed"],
    )
    def test_forward_combines_types_with_all_poolings(self, pooling) -> None:
        """All four pooling variants compose correctly with mixed inputs."""
        enc = _build(
            continuous=("condition0",),
            categorical=("condition1",),
            pooling=pooling,
        ).eval()
        cond = _inputs(continuous=("condition0",), categorical=("condition1",))
        with torch.no_grad():
            out = enc(cond)
        assert out.shape == (BATCH, OUTPUT_DIM)
        # No NaN/Inf
        assert torch.isfinite(out).all()

    # ---- grads through both paths ----------------------------------------

    def test_both_paths_receive_gradients(self) -> None:
        """Backprop reaches both the categorical and continuous encoder
        parameters when both are used.
        """
        enc = _build(
            continuous=("condition0",),
            categorical=("condition1",),
            pooling=MeanPoolingConfig(),
        )
        cond = _inputs(continuous=("condition0",), categorical=("condition1",))
        enc(cond).sum().backward()

        cat_params = list(enc._set_encoder["input_layers"]["categorical"]["condition1"].parameters())
        cont_params = list(enc._set_encoder["input_layers"]["continuous"]["condition0"].parameters())
        proj_params = list(enc._set_encoder["projection"]["condition1"].parameters())
        out_params = list(enc._set_encoder["output_layer"].parameters())

        assert all(p.grad is not None for p in cat_params)
        assert all(p.grad is not None for p in cont_params)
        assert all(p.grad is not None for p in proj_params)
        assert all(p.grad is not None for p in out_params)
