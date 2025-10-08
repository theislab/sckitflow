import pytest
from sc_flow.backends.torch.methods import FlowMatching


class TestTorchFlowMatching:

    def test_forward_call(self):
        fm = FlowMatching(vf=vf, )


    @pytest.mark.parametrize("probability_path", ["constant-noise-linear-gaussian", "cnlg-pp", "ld-pp"])    
    def test_probability_paths(self, probability_path: str):
        fm = FlowMatching(vf=vf, probability_path=probability_path)
        fm.step_fn(batch=batch)
        