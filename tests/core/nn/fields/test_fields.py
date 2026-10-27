"""Tests for :class:`NeuralVelocity` built from :class:`NeuralVelocityConfig`."""

import pandas as pd
import torch

from sckitflow.core.nn import MLPConfig, MLPEncoderConfig, NeuralVelocityConfig
from sckitflow.data._dims import DataDimensions


def _dims(**kwargs) -> DataDimensions:
    fields = dict.fromkeys(
        (
            "condition_reps_dims",
            "condition_continuous_dims",
            "response_reps_dims",
            "response_continuous_dims",
            "groups_reps_dims",
            "source_lin_dim",
            "source_quad_dim",
            "target_lin_dim",
            "target_quad_dim",
        )
    )
    return DataDimensions(state_dim=5, feature_names=pd.Index(list("abcde")), **{**fields, **kwargs})


def test_default_field_sees_the_scale_and_offset_of_the_state():
    torch.manual_seed(0)
    module = NeuralVelocityConfig().build(_dims())
    t, x = torch.rand(8), torch.randn(8, 5)
    assert not torch.allclose(module(t, x), module(t, 3 * x + 7))


def test_sources_sharing_the_state_are_encoded_from_it():
    """``source_state_dim=None`` means the sources share the targets' state, so the encoder reads ``state_dim``."""
    torch.manual_seed(0)
    cfg = NeuralVelocityConfig(source_encoder=MLPEncoderConfig(mlp=MLPConfig(), output_dim=4))
    module = cfg.build(_dims())
    t, x = torch.rand(8), torch.randn(8, 5)
    assert not torch.allclose(module(t, x, source=torch.randn(8, 5)), module(t, x, source=torch.randn(8, 5)))
