"""Tests for :mod:`sckitflow._samplers`.

``SamplerConfig`` mirrors ``CouplingConfig``: ``fn`` is a ``ClassVar`` so the
class itself is what the modelbinder registry serializes, and ``build`` returns the
raw callable (or a ``partial`` when a subclass adds fields). These tests lock
in that contract — portable round-trip, deterministic draws via the passed
generator, device / dtype forwarding, and the partial-with-fields pattern.
"""

import json

import pytest
import torch
from modelbinder.registry import component

from sckitflow._samplers import (
    IsotropicGaussianNoiseSamplerConfig,
    SamplerConfig,
    TimeSamplerConfig,
    UniformTimeSamplerConfig,
    _standard_normal,
    _uniform,
)


def gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


# -------------------- Family base --------------------
def test_base_sampler_config_is_not_buildable() -> None:
    """The family base leaves ``fn`` unbound; ``build`` cannot return a callable."""
    with pytest.raises((AttributeError, TypeError, ValueError)):
        SamplerConfig().build()


# -------------------- Concrete implementations --------------------
def test_uniform_time_sampler_builds_to_torch_rand() -> None:
    assert UniformTimeSamplerConfig().build() is _uniform


def test_isotropic_gaussian_noise_sampler_builds_to_torch_randn() -> None:
    assert IsotropicGaussianNoiseSamplerConfig().build() is _standard_normal


# -------------------- Draw semantics --------------------
def test_uniform_sampler_produces_values_in_unit_interval() -> None:
    out = UniformTimeSamplerConfig().build()((10_000,), generator=gen())
    assert out.shape == (10_000,)
    assert torch.all(out >= 0) and torch.all(out < 1)
    # Rough check that it's actually uniform-ish
    assert 0.45 < out.mean().item() < 0.55


def test_noise_sampler_produces_standard_normal_like_samples() -> None:
    out = IsotropicGaussianNoiseSamplerConfig().build()((20_000,), generator=gen())
    assert out.shape == (20_000,)
    assert out.mean().abs().item() < 0.05
    assert abs(out.std().item() - 1.0) < 0.05


# -------------------- Generator-only determinism --------------------
@pytest.mark.parametrize(
    "config_cls",
    [UniformTimeSamplerConfig, IsotropicGaussianNoiseSamplerConfig],
)
def test_sampler_depends_only_on_generator(config_cls: type[SamplerConfig]) -> None:
    """Two calls with the same ``generator`` seed and different global seeds agree."""
    sampler = config_cls().build()
    torch.manual_seed(123)
    a = sampler((10,), generator=gen(0))
    torch.manual_seed(999)
    b = sampler((10,), generator=gen(0))
    torch.testing.assert_close(a, b)

    # A different generator seed must produce a different draw
    c = sampler((10,), generator=gen(1))
    assert not torch.equal(a, c)


# -------------------- Device / dtype forwarding --------------------
@pytest.mark.parametrize(
    "config_cls",
    [UniformTimeSamplerConfig, IsotropicGaussianNoiseSamplerConfig],
)
def test_sampler_honours_device_and_dtype(config_cls: type[SamplerConfig]) -> None:
    sampler = config_cls().build()
    out = sampler(
        (5,),
        generator=gen(),
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert out.device.type == "cpu"
    assert out.dtype == torch.float64


# -------------------- Serialization --------------------
@pytest.mark.parametrize(
    "config_cls",
    [UniformTimeSamplerConfig, IsotropicGaussianNoiseSamplerConfig],
)
def test_sampler_config_round_trips(config_cls: type[SamplerConfig]) -> None:
    cfg = config_cls()
    rebuilt = config_cls.from_spec(cfg.to_spec())
    assert rebuilt == cfg


@pytest.mark.parametrize(
    "config_cls",
    [UniformTimeSamplerConfig, IsotropicGaussianNoiseSamplerConfig],
)
def test_sampler_config_spec_is_json_serializable(config_cls: type[SamplerConfig]) -> None:
    """The whole point of the config object: JSON, not a Python callable."""
    json.dumps(config_cls().to_spec())  # must not raise


# -------------------- Partial-with-fields pattern --------------------
def _scaled_ones(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    device: torch.types.Device = None,
    dtype: torch.dtype | None = None,
    scale: float = 1.0,
) -> torch.Tensor:
    return scale * torch.ones(shape, device=device, dtype=dtype)


@component("time_sampler.scaled_ones")
class ScaledOnesSamplerConfig(TimeSamplerConfig):
    """Test-only sampler that bakes a scale into the built callable."""

    fn = _scaled_ones
    scale: float = 2.0


def test_subclass_fields_are_baked_into_the_built_callable() -> None:
    sampler = ScaledOnesSamplerConfig(scale=3.5).build()
    out = sampler((3,), generator=gen())
    torch.testing.assert_close(out, torch.full((3,), 3.5))


def test_subclass_fields_round_trip() -> None:
    cfg = ScaledOnesSamplerConfig(scale=3.5)
    rebuilt = ScaledOnesSamplerConfig.from_spec(cfg.to_spec())
    assert rebuilt.scale == 3.5


def test_subclass_with_fields_produces_a_partial_not_the_fn() -> None:
    """With fields, ``build`` returns a ``functools.partial`` — not ``fn`` itself."""
    from functools import partial

    built = ScaledOnesSamplerConfig(scale=3.5).build()
    assert isinstance(built, partial)
    assert built.keywords == {"scale": 3.5}
    assert built.func is _scaled_ones
