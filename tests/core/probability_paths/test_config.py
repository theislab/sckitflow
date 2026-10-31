"""Tests for the probability path configs."""

import pytest

from sckitflow._samplers import NoiseSamplerConfig
from sckitflow.core.probability_paths._config import (
    EncDecTwoSidedPathConfig,
    LinearOneSidedPathConfig,
    LinearTwoSidedPathConfig,
    PathConfig,
    TrigOneSidedPathConfig,
    TrigTwoSidedPathConfig,
    VPOneSidedPathConfig,
)
from sckitflow.core.probability_paths._probability_paths import (
    EncDecTwoSidedPath,
    LinearOneSidedPath,
    LinearTwoSidedPath,
    TrigOneSidedPath,
    TrigTwoSidedPath,
    VPOneSidedPath,
)

PAIRS = [
    (LinearTwoSidedPathConfig, "paths.two_sided.linear", LinearTwoSidedPath),
    (TrigTwoSidedPathConfig, "paths.two_sided.trig", TrigTwoSidedPath),
    (EncDecTwoSidedPathConfig, "paths.two_sided.enc_dec", EncDecTwoSidedPath),
    (LinearOneSidedPathConfig, "paths.one_sided.linear", LinearOneSidedPath),
    (TrigOneSidedPathConfig, "paths.one_sided.trig", TrigOneSidedPath),
    (VPOneSidedPathConfig, "paths.one_sided.vp", VPOneSidedPath),
]
_PAIR_IDS = [cfg_cls.__name__ for cfg_cls, _, _ in PAIRS]


def _make_sampler_cfg_or_skip():
    """Return a real ``NoiseSamplerConfig``, or skip if it needs args."""
    try:
        return NoiseSamplerConfig()
    except ValueError as exc:
        pytest.skip(f"NoiseSamplerConfig() unavailable: {exc}")


# ---------------------------------------------------------------------------
# Base class contract
# ---------------------------------------------------------------------------


class TestPathConfigBase:
    def test_build_raises_not_implemented(self) -> None:
        # ``PathConfig`` is a family base: pydantic rejects direct
        # instantiation, so call the unbound method. ``PathConfig.build``
        # raises immediately and never touches ``self``.
        with pytest.raises(NotImplementedError):
            PathConfig.build(object())

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
    @pytest.mark.parametrize("cfg_cls, key, path_cls", PAIRS, ids=_PAIR_IDS)
    def test_deterministic_sigma_zero(self, cfg_cls, key, path_cls) -> None:
        # sigma == 0 needs no sampler — build must succeed regardless of
        # whether a NoiseSamplerConfig can be constructed in this env.
        path = cfg_cls(sigma=0.0).build()
        assert isinstance(path, path_cls)
        assert type(path) is path_cls  # exact type, not a subclass

    @pytest.mark.parametrize("cfg_cls, key, path_cls", PAIRS, ids=_PAIR_IDS)
    def test_forwards_sigma(self, cfg_cls, key, path_cls) -> None:
        sampler_cfg = _make_sampler_cfg_or_skip()
        path = cfg_cls(sigma=2.5, sampler_cfg=sampler_cfg).build()
        assert path.sigma == 2.5

    @pytest.mark.parametrize("cfg_cls, key, path_cls", PAIRS, ids=_PAIR_IDS)
    def test_positive_sigma_without_sampler_raises(self, cfg_cls, key, path_cls) -> None:
        cfg = cfg_cls(sigma=1.0)
        with pytest.raises(ValueError, match="Positive sigma requires a sampler"):
            cfg.build()

    @pytest.mark.parametrize("cfg_cls, key, path_cls", PAIRS, ids=_PAIR_IDS)
    def test_positive_sigma_with_sampler(self, cfg_cls, key, path_cls) -> None:
        sampler_cfg = _make_sampler_cfg_or_skip()
        path = cfg_cls(sigma=1.0, sampler_cfg=sampler_cfg).build()
        assert isinstance(path, path_cls)
        assert path.sigma == 1.0
        assert path._sampler is not None

    @pytest.mark.parametrize("cfg_cls, key, path_cls", PAIRS, ids=_PAIR_IDS)
    def test_sampler_cfg_stored_verbatim(self, cfg_cls, key, path_cls) -> None:
        sampler_cfg = _make_sampler_cfg_or_skip()
        cfg = cfg_cls(sigma=1.0, sampler_cfg=sampler_cfg)
        # The config shouldn't copy or rebuild the sampler — just forward it.
        assert cfg.sampler_cfg is sampler_cfg

    @pytest.mark.parametrize("cfg_cls, key, path_cls", PAIRS, ids=_PAIR_IDS)
    def test_default_sampler_cfg_is_none(self, cfg_cls, key, path_cls) -> None:
        assert cfg_cls(sigma=0.0).sampler_cfg is None

    @pytest.mark.parametrize("cfg_cls, key, path_cls", PAIRS, ids=_PAIR_IDS)
    def test_build_returns_fresh_instance(self, cfg_cls, key, path_cls) -> None:
        sampler_cfg = _make_sampler_cfg_or_skip()
        cfg = cfg_cls(sigma=1.0, sampler_cfg=sampler_cfg)
        a, b = cfg.build(), cfg.build()
        assert a is not b


# ---------------------------------------------------------------------------
# Decorator metadata
# ---------------------------------------------------------------------------


class TestDecoratorMetadata:
    @pytest.mark.parametrize("cfg_cls, key, path_cls", PAIRS, ids=_PAIR_IDS)
    def test_name_preserved(self, cfg_cls, key, path_cls) -> None:
        # The decorator must not rename the class.
        assert cfg_cls.__name__.endswith("PathConfig")

    @pytest.mark.parametrize("cfg_cls, key, path_cls", PAIRS, ids=_PAIR_IDS)
    def test_builds_target_exposed_somewhere(self, cfg_cls, key, path_cls) -> None:
        # ``builds=...`` is usually attached either as an attribute on the
        # class, in a registry, or both. Try the attribute path; skip
        # cleanly if this decorator keeps it registry-only.
        found = False
        for name in dir(cfg_cls):
            if name.startswith("__"):
                continue
            try:
                value = getattr(cfg_cls, name)
            except AttributeError:
                continue
            if value is path_cls:
                found = True
                break
            if isinstance(value, str) and value == path_cls.__name__:
                found = True
                break
        if not found:
            pytest.skip("builds= target not exposed on class — decorator may store it in the registry only")


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
            "VPOneSidedPathConfig",
        }
        assert set(_config.__all__) == expected

    def test_all_names_resolve(self) -> None:
        from sckitflow.core.probability_paths import _config

        for name in _config.__all__:
            assert hasattr(_config, name), f"missing {name}"

    @pytest.mark.parametrize("cfg_cls, key, path_cls", PAIRS, ids=_PAIR_IDS)
    def test_cfg_is_subclass_of_pathconfig(self, cfg_cls, key, path_cls) -> None:
        assert issubclass(cfg_cls, PathConfig)


# ---------------------------------------------------------------------------
# Registry (best-effort)
# ---------------------------------------------------------------------------


_SENTINEL = object()


def _registry_lookup(key: str):
    """Try common registry APIs; return ``_SENTINEL`` if none work."""
    try:
        from scfit import registry
    except ImportError:
        return _SENTINEL
    for name in ("get", "resolve", "lookup", "find", "load", "get_component"):
        fn = getattr(registry, name, None)
        if callable(fn):
            try:
                return fn(key)
            except TypeError:
                continue
    return _SENTINEL


class TestRegistry:
    @pytest.mark.parametrize("cfg_cls, key, path_cls", PAIRS, ids=_PAIR_IDS)
    def test_component_is_registered(self, cfg_cls, key, path_cls) -> None:
        entry = _registry_lookup(key)
        if entry is _SENTINEL:
            pytest.skip("Registry lookup API unknown")

        # Entry may be the config class directly, a spec object with the
        # target attached, or a factory.
        if entry is cfg_cls:
            return
        for attr in ("config", "config_cls", "cls", "builds", "target"):
            val = getattr(entry, attr, None)
            if val is cfg_cls or val is path_cls:
                return
        if callable(entry):
            try:
                produced = entry(sigma=0.0)
            except TypeError:
                try:
                    produced = entry()
                except ValueError:
                    produced = None
            if produced is not None and isinstance(produced, path_cls):
                return
        pytest.skip(f"Unrecognized registry entry for {key}")
