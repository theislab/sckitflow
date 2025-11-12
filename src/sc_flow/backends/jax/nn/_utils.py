from flax.core import FrozenDict

from sc_flow._types import LayersDict
from sc_flow._utils import verify_fn_kwargs_dictionary
from sc_flow.backends.jax.nn._modules import MLP, BaseModule

__all__ = ["init_module_from_dict"]


def init_module_from_dict(
    layers_dict: LayersDict,
    input_dim: int | None = None,
    output_dim: int | None = None,
) -> BaseModule:
    """Initilizes a base module from the specified settings.

    The type is specified by the `"layer_type"` key and falls back to
    :class: `sc_flow.backends.torch.nn.MLP` by default.

    * When initializing MLPs the configuration dictionary should also contain the `"input_dim"`
        and `"output_dim"` keys that are needed for the initialization of the module.
        If the respective arguments are passed instead, the keys from the dictionary will be removed and
        ignored.

    :param layers_dict: The dictionary containing the configurations for the module.
    :type layers_dict: class: `LayersDict`

    :param input_dim: (Optional) Input dimension for the module.
    :type input_dim: class: `int | None`

    :param output_dim: (Optional) Output dimension for the module.
    :type output_dim: class: `int | None`
    """

    if isinstance(layers_dict, FrozenDict):
        layers_dict = dict(layers_dict)
    else:
        layers_dict = layers_dict.copy()

    # retrieving layer type
    layer_type = layers_dict.pop("layer_type", "mlp")

    # initializing mlp
    if layer_type == "mlp":
        # retrieving input dim
        if input_dim is None:
            # sanity check
            if "input_dim" not in layers_dict.keys():
                msg = 'Layers dictionary should contain the "input_dim" key.'
                raise KeyError(msg)
            # retrieving key
            input_dim = layers_dict.pop("input_dim")
        else:
            # discarding key
            layers_dict.pop("input_dim", None)
        # retrieving output dim
        if output_dim is None:
            # sanity check
            if "output_dim" not in layers_dict.keys():
                msg = 'Layers dictionary should contain the "output_dim" key.'
                raise KeyError(msg)
            # retrieving key
            output_dim = layers_dict.pop("output_dim")
        else:
            # discarding key
            layers_dict.pop("output_dim", None)

        # sanity check
        if not isinstance(input_dim, int):
            msg = f"Argument `input_dim` is expected to be an `int`, {type(input_dim)} found."
            raise TypeError(msg)
        if not isinstance(input_dim, int):
            msg = f"Argument `input_dim` is expected to be an `int`, {type(input_dim)} found."
            raise TypeError(msg)

        # verify keyword arguments
        verify_fn_kwargs_dictionary(MLP.__init__, layers_dict)
        return MLP(input_dim, output_dim, **layers_dict)
    else:
        msg = f'{layer_type=} is not supported, possible choices are `["mlp"]`'
        raise ValueError(msg)
