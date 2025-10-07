import pytest
from sc_flow.backends.jax.methods import FlowMatching

class TestFlowMatching:


    @pytest.mark.parametrize("foo", "bar")
    def test_simple_run(self):
        fm = FlowMatching(vf=vf)
        