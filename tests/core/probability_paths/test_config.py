"""Tests for the probability path configs."""

import pytest
from scfit.registry import config_of, parse, to_spec

from sckitflow import IsotropicGaussianNoiseSamplerConfig
from sckitflow.core.probability_paths._config import (
    EncDecTwoSidedPathConfig,
    LinearOneSidedPathConfig,
    LinearTwoSidedPathConfig,
    PathConfig,
    TrigOneSidedPathConfig,
    TrigTwoSidedPathConfig,
)
from sckitflow.core.probability_paths._probability_paths import (
    EncDecTwoSidedPath,
    LinearOneSidedPath,
    LinearTwoSidedPath,
    TrigOneSidedPath,
    TrigTwoSidedPath,
)

PAIRS = [
    (LinearTwoSidedPathConfig, "paths.two_sided.linear", LinearTwoSidedPath),
    (TrigTwoSidedPathConfig, "paths.two_sided.trig", TrigTwoSidedPath),
    (EncDecTwoSidedPathConfig, "paths.two_sided.enc_dec", EncDecTwoSidedPath),
    (LinearOneSidedPathConfig, "paths.one_sided.linear", LinearOneSidedPath),
    (TrigOneSidedPathConfig, "paths.one_sided.trig", TrigOneSidedPath),
]
_PAIR_IDS = [cfg_cls.__name__ for cfg_cls, _, _ in PAIRS]
TWO_SIDED_PAIRS, ONE_SIDED_PAIRS = PAIRS[:3], PAIRS[3:]


def _valid_cfg(cfg_cls):
    """A buildable config: one-sided paths take ``sigma == 1`` and no sampler, two-sided ones a sampler."""
    if cfg_cls in {c for c, _, _ in ONE_SIDED_PAIRS}:
        return cfg_cls(sigma=1.0)
    return cfg_cls(sigma=1.0, sampler_cfg=IsotropicGaussianNoiseSamplerConfig())


# ---------------------------------------------------------------------------
# Base class contract
# ---------------------------------------------------------------------------


class TestPathConfigBase:
    def test_sigma_and_sampler_cfg_fields(self) -> None:
        # No instance required: inspect the pydantic field declarations.
        fields = PathConfig.model_fields
        assert "sigma" in fields
        assert "sampler_cfg" in fields
        assert fields["sigma"].is_required()
        assert fields["sampler_cfg"].default is None

    def test_pathconfig_is_family_base(self) -> None:
        # Documents the constraint the previous two tests work around.
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="family base"):
            PathConfig(sigma=0.0)


# ---------------------------------------------------------------------------
# build() behaviour
# ---------------------------------------------------------------------------


class TestConfigBuild:
    @pytest.mark.parametrize("cfg_cls, key, path_cls", TWO_SIDED_PAIRS, ids=_PAIR_IDS[:3])
    def test_deterministic_sigma_zero(self, cfg_cls, key, path_cls) -> None:
        # sigma == 0 needs no sampler.
        path = cfg_cls(sigma=0.0).build()
        assert isinstance(path, path_cls)
        assert type(path) is path_cls  # exact type, not a subclass

    @pytest.mark.parametrize("cfg_cls, key, path_cls", TWO_SIDED_PAIRS, ids=_PAIR_IDS[:3])
    def test_forwards_sigma(self, cfg_cls, key, path_cls) -> None:
        path = cfg_cls(sigma=2.5, sampler_cfg=IsotropicGaussianNoiseSamplerConfig()).build()
        assert path.sigma == 2.5

    @pytest.mark.parametrize("cfg_cls, key, path_cls", TWO_SIDED_PAIRS, ids=_PAIR_IDS[:3])
    def test_positive_sigma_without_sampler_raises(self, cfg_cls, key, path_cls) -> None:
        with pytest.raises(ValueError, match="Positive sigma requires a sampler"):
            cfg_cls(sigma=1.0).build()

    @pytest.mark.parametrize("cfg_cls, key, path_cls", PAIRS, ids=_PAIR_IDS)
    def test_valid_config_builds(self, cfg_cls, key, path_cls) -> None:
        path = _valid_cfg(cfg_cls).build()
        assert type(path) is path_cls
        assert path.sigma == 1.0

    @pytest.mark.parametrize("cfg_cls, key, path_cls", PAIRS, ids=_PAIR_IDS)
    def test_default_sampler_cfg_is_none(self, cfg_cls, key, path_cls) -> None:
        assert cfg_cls(sigma=0.0).sampler_cfg is None

    @pytest.mark.parametrize("cfg_cls, key, path_cls", PAIRS, ids=_PAIR_IDS)
    def test_build_returns_fresh_instance(self, cfg_cls, key, path_cls) -> None:
        cfg = _valid_cfg(cfg_cls)
        a, b = cfg.build(), cfg.build()
        assert a is not b


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    @pytest.mark.parametrize("cfg_cls, key, path_cls", PAIRS, ids=_PAIR_IDS)
    def test_config_of_path(self, cfg_cls, key, path_cls) -> None:
        assert config_of(path_cls) is cfg_cls

    @pytest.mark.parametrize("cfg_cls, key, path_cls", PAIRS, ids=_PAIR_IDS)
    def test_spec_round_trip(self, cfg_cls, key, path_cls) -> None:
        cfg = _valid_cfg(cfg_cls)
        spec = to_spec(cfg)
        assert spec["type"] == key
        assert parse(spec) == cfg


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


class TestModuleSurface:
    def test_all_exports(self) -> None:
        from sckitflow.core.probability_paths import _config

        expected = {
            "PathConfig",
            "LinearTwoSidedPathConfig",
            "TrigTwoSidedPathConfig",
            "EncDecTwoSidedPathConfig",
            "LinearOneSidedPathConfig",
            "TrigOneSidedPathConfig",
        }
        assert set(_config.__all__) == expected

    def test_all_names_resolve(self) -> None:
        from sckitflow.core.probability_paths import _config

        for name in _config.__all__:
            assert hasattr(_config, name), f"missing {name}"

    @pytest.mark.parametrize("cfg_cls, key, path_cls", PAIRS, ids=_PAIR_IDS)
    def test_cfg_is_subclass_of_pathconfig(self, cfg_cls, key, path_cls) -> None:
        assert issubclass(cfg_cls, PathConfig)
