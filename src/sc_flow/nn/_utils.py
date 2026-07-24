from sc_flow._types import LayersDict
from sc_flow.nn._modules import MLP

__all__ = ["init_module_from_dict"]


def _resolve_dim(layers_dict: LayersDict, key: str, value: int | None) -> int:
    """Resolve ``input_dim``/``output_dim`` from an explicit argument or the layer dict.

    An explicit ``value`` wins (the dict's key, if present, is discarded); otherwise the dim must be
    carried in ``layers_dict`` (e.g. per-realm input layers whose dims are part of the config).
    """
    if value is None:
        if key not in layers_dict:
            raise KeyError(f'Layers dictionary must contain "{key}" when it is not passed explicitly.')
        value = layers_dict.pop(key)
    else:
        layers_dict.pop(key, None)
    if not isinstance(value, int):
        raise TypeError(f"`{key}` is expected to be an int, {type(value).__name__} found.")
    return value


def init_module_from_dict(
    layers_dict: LayersDict,
    input_dim: int | None = None,
    output_dim: int | None = None,
) -> MLP:
    """Build an :class:`MLP` from a JSON layer-kwargs dict.

    ``input_dim``/``output_dim`` may be passed as arguments (from runtime context) or carried in
    ``layers_dict`` (dims-as-config); remaining keys are forwarded to :class:`MLP`. MLP is the only layer
    type — a second polymorphic layer family would be a discriminated ``ComponentSpec``, not a string
    branch here.
    """
    layers_dict = layers_dict.copy()
    input_dim = _resolve_dim(layers_dict, "input_dim", input_dim)
    output_dim = _resolve_dim(layers_dict, "output_dim", output_dim)
    return MLP(input_dim, output_dim, **layers_dict)
