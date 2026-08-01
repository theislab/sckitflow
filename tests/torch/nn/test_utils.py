import pytest

from sckitflow._types import LayersDict
from sckitflow.core.nn._modules import MLP
from sckitflow.core.nn._utils import init_module_from_dict

input_dim = 16
output_dim = 8


class TestNNUtils:
    @pytest.mark.parametrize(
        "layers_dict",
        [{}, {"input_dim": input_dim}, {"output_dim": output_dim}, {"input_dim": input_dim, "output_dim": output_dim}],
    )
    @pytest.mark.parametrize("input_dim", [None, input_dim])
    @pytest.mark.parametrize("output_dim", [None, output_dim])
    def test_init_module_from_dict(
        self,
        layers_dict: LayersDict,
        input_dim: int | None,
        output_dim: int | None,
    ) -> None:
        # missing input dimension
        if input_dim is None and "input_dim" not in layers_dict.keys():
            with pytest.raises(KeyError, match=r"Layers dictionary should contain the \"input_dim\" key\."):
                layer = init_module_from_dict(layers_dict, input_dim=input_dim, output_dim=output_dim)
            return None

        # missing output dimension
        if output_dim is None and "output_dim" not in layers_dict.keys():
            with pytest.raises(KeyError, match=r"Layers dictionary should contain the \"output_dim\" key\."):
                layer = init_module_from_dict(layers_dict, input_dim=input_dim, output_dim=output_dim)
            return None

        # correct initialization
        layer = init_module_from_dict(layers_dict, input_dim=input_dim, output_dim=output_dim)

        # check that is is of the correct type
        assert isinstance(layer, MLP)
