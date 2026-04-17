import pytest
import torch

from sc_flow import _constants
from sc_flow.backends.torch.methods import FlowMatching
from sc_flow.backends.torch.nn._vf import MLPVelocity
from sc_flow.backends.torch.probability_paths import LinearDiracProbabilityPath

STATE_DIM = 20

batch = {}
batch[_constants.SOURCE_STATE] = torch.randn((20, STATE_DIM))
batch[_constants.TARGET_STATE] = torch.randn((20, STATE_DIM))


vf = MLPVelocity(state_dim=STATE_DIM)


class TestTorchFlowMatching:
    @pytest.mark.parametrize("generate_from_noise", [True, False])
    def test_forward_call(self, generate_from_noise):
        if generate_from_noise:
            batch[_constants.SOURCE_STATE] = None
        else:
            batch[_constants.SOURCE_STATE] = torch.randn((20, STATE_DIM))
        fm = FlowMatching(
            vf=vf,
            time_sampler=torch.rand,
            probability_path=LinearDiracProbabilityPath(),
            generate_from_noise=generate_from_noise,
        )
        fm.step_fn(batch=batch)
