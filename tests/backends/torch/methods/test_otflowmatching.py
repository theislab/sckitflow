import pytest
import torch

from sc_flow import _constants
from sc_flow.backends.torch.methods import OTFlowMatching
from sc_flow.backends.torch.nn._vf import MLPUnconditionalVF
from sc_flow.backends.torch.probability_paths import LinearDiracProbabilityPath

STATE_DIM = 20

batch = {}
batch[_constants.SOURCE_STATE] = torch.randn((20, STATE_DIM))
batch[_constants.TARGET_STATE] = torch.randn((20, STATE_DIM))

match_fn = lambda x: (torch.rand(20), torch.rand(20))
vf = MLPUnconditionalVF(state_dim=STATE_DIM)


class TestTorchFlowMatching:
    @pytest.mark.parametrize("generate_from_noise", [True, False])
    def test_forward_call(self, generate_from_noise):
        if generate_from_noise:
            batch[_constants.SOURCE_STATE] = None
        else:
            batch[_constants.SOURCE_STATE] = torch.randn((20, STATE_DIM))
        otfm = OTFlowMatching(
            vf=vf,
            time_sampler=torch.rand,
            match_fn=match_fn,
            probability_path=LinearDiracProbabilityPath(),
            generate_from_noise=generate_from_noise,
        )
        otfm.step_fn(batch=batch)
