import pytest
import torch

from sc_flow import _constants
from sc_flow.backends.torch.methods import FlowMatching
from sc_flow.backends.torch.nn._vf import MLPUnconditionalVF

STATE_DIM = 20

batch = {}
batch[_constants.SOURCE_STATE] = torch.randn((20, STATE_DIM))
batch[_constants.TARGET_STATE] = torch.randn((20, STATE_DIM))


vf = MLPUnconditionalVF(state_dim=STATE_DIM)


class TestTorchFlowMatching:
    @pytest.mark.parametrize("generate_from_noise", [True, False])
    def test_forward_call(self, generate_from_noise):
        if generate_from_noise is True:
            batch[_constants.SOURCE_STATE] = None
        fm = FlowMatching(
            vf=vf,
            time_sampler=torch.rand,
            probability_path="constant-noise-linear-gaussian",
            generate_from_noise=generate_from_noise,
        )
        fm.step_fn(batch=batch)

    @pytest.mark.parametrize("generate_from_noise", [True, False])
    @pytest.mark.parametrize("probability_path", ["constant-noise-linear-gaussian", "cnlg-pp", "ld-pp"])
    def test_probability_paths(self, probability_path: str, generate_from_noise: bool):
        if generate_from_noise is True:
            batch[_constants.SOURCE_STATE] = None
        fm = FlowMatching(
            vf=vf, time_sampler=torch.rand, probability_path=probability_path, generate_from_noise=generate_from_noise
        )
        fm.step_fn(batch=batch)
