"""Tests for the probability path configs."""

import pydantic
import pytest
from scfit.registry import config_of, parse, to_spec

from sckitflow import IsotropicGaussianNoiseSamplerConfig
from sckitflow.core.probability_paths._config import EncDecPathConfig, LinearPathConfig, TrigPathConfig
from sckitflow.core.probability_paths._probability_paths import EncDecPath, LinearPath, TrigPath

PAIRS = [
    (LinearPathConfig, "probability_path.linear", LinearPath),
    (TrigPathConfig, "probability_path.trig", TrigPath),
    (EncDecPathConfig, "probability_path.enc_dec", EncDecPath),
]
_IDS = [cfg_cls.__name__ for cfg_cls, _, _ in PAIRS]
_GAUSS = IsotropicGaussianNoiseSamplerConfig()


@pytest.mark.parametrize(("cfg_cls", "key", "path_cls"), PAIRS, ids=_IDS)
def test_builds_its_path(cfg_cls, key, path_cls) -> None:
    path = cfg_cls(sigma=2.5, sampler_cfg=_GAUSS).build()
    assert type(path) is path_cls
    assert path.sigma == 2.5


@pytest.mark.parametrize("cfg_cls", [LinearPathConfig, TrigPathConfig])
def test_default_is_noiseless(cfg_cls) -> None:
    assert cfg_cls().build().sigma == 0.0


@pytest.mark.parametrize(("cfg_cls", "key", "path_cls"), PAIRS, ids=_IDS)
@pytest.mark.parametrize("sigma", [-1.0, float("inf"), float("nan")])
def test_invalid_sigma_rejected_at_config_time(cfg_cls, key, path_cls, sigma) -> None:
    with pytest.raises(pydantic.ValidationError):
        cfg_cls(sigma=sigma, sampler_cfg=_GAUSS)


@pytest.mark.parametrize(("cfg_cls", "key", "path_cls"), PAIRS, ids=_IDS)
def test_positive_sigma_without_sampler_rejected_at_config_time(cfg_cls, key, path_cls) -> None:
    with pytest.raises(pydantic.ValidationError, match="Positive sigma requires a sampler_cfg"):
        cfg_cls(sigma=1.0)


@pytest.mark.parametrize("kwargs", [{}, {"sigma": 0.0}])
def test_enc_dec_needs_positive_sigma(kwargs) -> None:
    with pytest.raises(pydantic.ValidationError, match="sigma"):
        EncDecPathConfig(**kwargs)


@pytest.mark.parametrize(("cfg_cls", "key", "path_cls"), PAIRS, ids=_IDS)
def test_registry(cfg_cls, key, path_cls) -> None:
    assert config_of(path_cls) is cfg_cls
    cfg = cfg_cls(sigma=1.0, sampler_cfg=_GAUSS)
    spec = to_spec(cfg)
    assert spec["type"] == key
    assert parse(spec) == cfg
