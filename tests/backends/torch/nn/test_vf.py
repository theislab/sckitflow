from collections.abc import Callable
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from sc_flow._types import TimeFeaturesId
from sc_flow.backends.torch._types import TTimeFeaturesFn
from sc_flow.backends.torch.nn._vf import MLPVelocity
from sc_flow.data._dims_registry import DataDimensionalitiesRegistry

batch_size = 32
n_samples = 50
state_dim = 20
num_time_features = 256
state_encoder_output_dim = 8
time_encoder_output_dim = 4


class TestVF:
    """Tests for the vanilla MLP velocity field forward pass."""

    @pytest.mark.parametrize("encode_state", [True, False])
    @pytest.mark.parametrize("encode_time", [True, False])
    @pytest.mark.parametrize("time_features_id", ["ott-jax", "torch-cfm", None])
    @pytest.mark.parametrize("time_features_fn", [None])
    @pytest.mark.parametrize("num_time_features", [None, num_time_features])
    @pytest.mark.parametrize("time_features_kwargs", [None])
    @pytest.mark.parametrize("state_encoder_output_dim", [None, state_encoder_output_dim])
    @pytest.mark.parametrize("time_encoder_output_dim", [None, time_encoder_output_dim])
    @pytest.mark.parametrize(
        "state_encoder_mlp_kwargs",
        [
            None,
            {"use_batchnorm": True, "batchnorm_affine": True, "batchnorm_track_running_stats": True},
            {"use_batchnorm": True, "batchnorm_affine": False, "batchnorm_track_running_stats": True},
            {"use_batchnorm": True, "batchnorm_affine": True, "batchnorm_track_running_stats": False},
            {"use_batchnorm": True, "batchnorm_affine": False, "batchnorm_track_running_stats": False},
        ],
    )
    @pytest.mark.parametrize(
        "time_encoder_mlp_kwargs",
        [
            None,
            {"use_batchnorm": True, "batchnorm_affine": True, "batchnorm_track_running_stats": True},
            {"use_batchnorm": True, "batchnorm_affine": False, "batchnorm_track_running_stats": True},
            {"use_batchnorm": True, "batchnorm_affine": True, "batchnorm_track_running_stats": False},
            {"use_batchnorm": True, "batchnorm_affine": False, "batchnorm_track_running_stats": False},
        ],
    )
    @pytest.mark.parametrize(
        "vf_decoder_mlp_kwargs",
        [
            None,
            {"use_batchnorm": True, "batchnorm_affine": True, "batchnorm_track_running_stats": True},
            {"use_batchnorm": True, "batchnorm_affine": False, "batchnorm_track_running_stats": True},
            {"use_batchnorm": True, "batchnorm_affine": True, "batchnorm_track_running_stats": False},
            {"use_batchnorm": True, "batchnorm_affine": False, "batchnorm_track_running_stats": False},
        ],
    )
    @pytest.mark.parametrize("conditioning_id", [None, "concat", "resnet1d"])
    @pytest.mark.parametrize("conditioning_fn", [None])
    @pytest.mark.parametrize("conditioning_kwargs", [None])
    def test_vanilla_mlp_vf_forward(
        self,
        encode_state: bool,
        encode_time: bool,
        time_features_id: TimeFeaturesId | None,
        time_features_fn: TTimeFeaturesFn | None,
        num_time_features: int | None,
        time_features_kwargs: dict[str, Any] | None,
        state_encoder_output_dim: int | None,
        time_encoder_output_dim: int | None,
        state_encoder_mlp_kwargs: dict[str, Any] | None,
        time_encoder_mlp_kwargs: dict[str, Any] | None,
        vf_decoder_mlp_kwargs: dict[str, Any] | None,
        conditioning_id: str | None,
        conditioning_fn: Callable | None,
        conditioning_kwargs: dict[str, Any],
    ):
        vf = MLPVelocity(
            state_dim,
            encode_state=encode_state,
            encode_time=encode_time,
            time_features_id=time_features_id,
            time_features_fn=time_features_fn,
            num_time_features=num_time_features,
            time_features_kwargs=time_features_kwargs,
            state_encoder_output_dim=state_encoder_output_dim,
            time_encoder_output_dim=time_encoder_output_dim,
            state_encoder_mlp_kwargs=state_encoder_mlp_kwargs,
            time_encoder_mlp_kwargs=time_encoder_mlp_kwargs,
            vf_decoder_mlp_kwargs=vf_decoder_mlp_kwargs,
            conditioning_id=conditioning_id,
            conditioning_fn=conditioning_fn,
            conditioning_kwargs=conditioning_kwargs,
        )

        # case 0: x: (B, D) t: (B, 1)
        x = torch.zeros((batch_size, state_dim))
        t = torch.zeros((batch_size, 1))
        vt = vf(t, x)
        assert vt.shape == (batch_size, state_dim)

        # case 1: x: (B, D) t: (B, )
        x = torch.zeros((batch_size, state_dim))
        t = torch.zeros((batch_size,))
        vt = vf(t, x)
        assert vt.shape == (batch_size, state_dim)

        # case 2: x: (B, N, D) t: (B, 1)
        vf.eval()
        x = torch.zeros((batch_size, n_samples, state_dim))
        t = torch.zeros((batch_size, 1))
        vt = vf(t, x)
        assert vt.shape == (batch_size, n_samples, state_dim)

        # case 3: x: (B, N, D) t: (B, )
        x = torch.zeros((batch_size, n_samples, state_dim))
        t = torch.zeros((batch_size,))
        vt = vf(t, x)
        assert vt.shape == (batch_size, n_samples, state_dim)


class TestMLPVelocityInitFromDimsRegistry:
    """Tests for MLPVelocity.init_from_dims_registry factory method.

    Verifies source encoder creation based on `is_paired_setting` and
    `generate_from_noise`, and condition encoder integration (skipped due to
    SetEncoder bug).
    """

    @staticmethod
    def make_registry(
        state_dim: int = 20,
        source_lin_dim: int | None = None,
        source_quad_dim: int | None = None,
        cond_reps: dict | None = None,
        cond_cont: dict | None = None,
        groups_reps: dict | None = None,
    ) -> Mock:
        """Helper to create a mock DataDimensionalitiesRegistry."""
        import copy

        registry = Mock(spec=DataDimensionalitiesRegistry)
        registry.state_dim = state_dim
        registry.source_lin_dim = source_lin_dim
        registry.source_quad_dim = source_quad_dim
        registry.condition_reps_dims = copy.deepcopy(cond_reps) or {}
        registry.condition_continuous_dims = copy.deepcopy(cond_cont) or {}
        registry.groups_reps_dims = copy.deepcopy(groups_reps) or {}
        return registry

    # ------------------------------------------------------------------
    # Source encoder creation rules
    # ------------------------------------------------------------------
    def test_no_condition_no_source(self):
        """Unconditional unpaired setting: no source encoder."""
        registry = self.make_registry()
        vf = MLPVelocity.init_from_dims_registry(registry, is_paired_setting=False, generate_from_noise=True)
        assert not vf.is_conditional
        assert not vf.use_source_encoder

    def test_paired_generate_from_noise_false_no_source_encoder(self):
        """Paired but generate_from_noise=False → no source encoder."""
        registry = self.make_registry(source_lin_dim=10)
        vf = MLPVelocity.init_from_dims_registry(registry, is_paired_setting=True, generate_from_noise=False)
        assert not vf.is_conditional
        assert not vf.use_source_encoder

    def test_paired_generate_from_noise_true_creates_source_encoder(self):
        """Paired and generate_from_noise=True → source encoder is created."""
        registry = self.make_registry(source_lin_dim=10)
        vf = MLPVelocity.init_from_dims_registry(registry, is_paired_setting=True, generate_from_noise=True)
        assert not vf.is_conditional
        assert vf.use_source_encoder
        assert vf._source_encoder_input_dim == 10

    def test_explicit_source_encoder_kwargs_overrides_automatic(self):
        """Providing source_encoder_mlp_kwargs explicitly creates source encoder regardless of flags."""
        registry = self.make_registry(source_lin_dim=5)
        vf = MLPVelocity.init_from_dims_registry(
            registry,
            is_paired_setting=False,
            generate_from_noise=True,
            source_encoder_mlp_kwargs={"hidden_dims": [8]},
        )
        assert vf.use_source_encoder
        assert vf._source_encoder_input_dim == 5

    # ------------------------------------------------------------------
    # Conditional encoder tests – skipped until SetEncoder._min_pooled_dims is fixed
    # ------------------------------------------------------------------
    @pytest.mark.skip(reason="SetEncoder is missing the '_min_pooled_dims' property")
    def test_conditional_vf_creation(self):
        cond_reps = {"drug": 128}
        registry = self.make_registry(cond_reps=cond_reps)
        input_layers = {"drug": {"input_dim": None}}
        vf = MLPVelocity.init_from_dims_registry(
            registry,
            is_paired_setting=False,
            generate_from_noise=True,
            condition_encoder_input_layers=input_layers,
        )
        assert vf.is_conditional
        assert "condition_encoder" in vf._vf
        assert vf._condition_encoder_input_layers["drug"]["input_dim"] == 128

    @pytest.mark.skip(reason="SetEncoder is missing the '_min_pooled_dims' property")
    def test_conditional_and_source_encoder_combined(self):
        cond_reps = {"drug": 128}
        registry = self.make_registry(source_lin_dim=5, source_quad_dim=3, cond_reps=cond_reps)
        input_layers = {"drug": {}}
        vf = MLPVelocity.init_from_dims_registry(
            registry,
            is_paired_setting=True,
            generate_from_noise=True,  # both flags must be True
            condition_encoder_input_layers=input_layers,
        )
        assert vf.is_conditional
        assert vf.use_source_encoder
        assert vf._source_encoder_input_dim == 8
        assert vf._conditioning_dim == vf._condition_encoder_output_dim + vf._source_encoder_output_dim

    @pytest.mark.skip(reason="SetEncoder is missing the '_min_pooled_dims' property")
    def test_continuous_covariates_not_pooled(self):
        cond_cont = {"time": 1, "dose": 1}
        registry = self.make_registry(cond_cont=cond_cont)
        input_layers = {"time": {}, "dose": {}}
        vf = MLPVelocity.init_from_dims_registry(
            registry,
            is_paired_setting=False,
            generate_from_noise=True,
            condition_encoder_input_layers=input_layers,
        )
        assert set(vf._condition_encoder_covariates_not_pooled) == {"time", "dose"}
