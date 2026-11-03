"""The potential components a surrogate's spec can name."""

import torch

from sckitflow.core.surrogate import CrossEntropyPotentialConfig, PotentialConfig, SquaredErrorPotentialConfig


def test_squared_error_sums_over_response_dimensions():
    y, ystar = torch.tensor([[1.0, 2.0], [0.0, 0.0]]), torch.tensor([1.0, 0.0])
    torch.testing.assert_close(SquaredErrorPotentialConfig().build()(y, ystar), torch.tensor([4.0, 1.0]))


def test_cross_entropy_is_minimal_at_the_target_and_finite_on_empty_classes():
    ystar = torch.tensor([0.7, 0.3, 0.0])
    y = torch.tensor([[0.7, 0.3, 0.0], [0.3, 0.7, 0.0], [0.0, 0.5, 0.5]])
    out = CrossEntropyPotentialConfig().build()(y, ystar)
    assert out[0] < out[1] < out[2]
    assert torch.isfinite(out).all()


def test_specs_round_trip_through_their_family():
    for cfg in (SquaredErrorPotentialConfig(), CrossEntropyPotentialConfig()):
        assert PotentialConfig.from_spec(cfg.to_spec()) == cfg


def test_built_potentials_pickle():
    """Module-level functions bound with `partial`, so a surrogate holding them can be pickled."""
    import pickle

    for built in (SquaredErrorPotentialConfig().build(), CrossEntropyPotentialConfig().build()):
        assert pickle.loads(pickle.dumps(built)).func is built.func
