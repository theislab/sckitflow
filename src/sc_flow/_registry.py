"""The uniform component registry + portable-spec (de)serialization for sc_flow.

One :class:`Component` base for every registrable/portable thing — sub-components, top-level
architectures, and objectives all use this same pattern. A concrete config is a plain ``@dataclass`` that
opts in with two class-header kwargs (``type_id=``, ``version=``/``versions=``) and owns
``build(self, context) -> runtime``. That single line auto-registers it (no decorator). The
``{type, version, config}`` envelope, registry dispatch, version check, family check, and unknown-field
rejection all live here, written once; leaf fields go through ``cattrs`` (so OmegaConf ``ListConfig`` /
``DictConfig`` values coerce natively).

Destined for ``scfit`` as the shared foundation once the neutral core moves there; kept in ``sc_flow`` for
now. See ``docs`` / the design decision for the rationale (no pydantic/msgspec: this is a public foundation
that downstream packages subclass, so keep the dependency surface minimal and the on-disk format stable).
"""

from __future__ import annotations

import dataclasses
import typing
from collections.abc import Mapping
from typing import Any, ClassVar, get_args, get_type_hints

import cattrs

__all__ = ["Component", "PortabilityError", "to_spec", "parse", "build", "register_live"]

_REGISTRY: dict[str, type["Component"]] = {}
_converter = cattrs.Converter(forbid_extra_keys=True)  # loud on a typo'd LEAF field
_hints_cache: dict[type, dict[str, Any]] = {}


class PortabilityError(Exception):
    """Raised when a config holding a live runtime instance is asked for its portable spec."""


class Component:
    """Base for every registrable/portable config.

    A concrete subclass opts in by passing ``type_id=`` (and optionally ``version=`` / ``versions=``) in
    its class header — that single line auto-registers it. A subclass with **no** ``type_id`` is an
    *abstract family base* (e.g. ``Objective``, ``Combiner``) and is intentionally left unregistered so it
    can be used as the ``expected`` family in :func:`parse` / :meth:`from_spec`.

    Every concrete config owns ``build(self, context) -> runtime``; ``__post_init__`` validation, derived
    ``@property`` sizing, and construction all live on the one class.
    """

    __type_id__: ClassVar[str]
    __version__: ClassVar[int]  # stamped when WRITING a spec
    __versions__: ClassVar[frozenset[int]]  # accepted set when READING a spec

    def __init_subclass__(
        cls,
        *,
        type_id: str | None = None,
        version: int = 1,
        versions: tuple[int, ...] | None = None,
        **kw: Any,
    ) -> None:
        super().__init_subclass__(**kw)
        if type_id is None:
            return  # abstract family base — not portable on its own
        if not isinstance(type_id, str) or not type_id:
            raise TypeError("type_id must be a non-empty string.")
        accepted = frozenset(versions) if versions is not None else frozenset({version})
        if version not in accepted or any((not isinstance(v, int)) or isinstance(v, bool) or v <= 0 for v in accepted):
            raise ValueError(f"{type_id!r}: bad version/versions ({version}, {sorted(accepted)}).")
        existing = _REGISTRY.get(type_id)
        if existing is not None and existing is not cls:
            raise ValueError(f"type_id {type_id!r} already registered to {existing.__name__}.")
        cls.__type_id__, cls.__version__, cls.__versions__ = type_id, version, accepted
        _REGISTRY[type_id] = cls

    def build(self, context: Any = None) -> Any:  # overridden by concrete configs
        raise NotImplementedError(f"{type(self).__name__} must implement build(self, context).")

    def to_spec(self) -> dict[str, Any]:
        """This config as a portable ``{type, version, config}`` dict. Raises on live instances."""
        return to_spec(self)

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> Any:
        """Parse a spec into a config, enforcing that it is a ``cls`` (family-scoped, typed entry).

        Replaces the per-family ``validate_<family>_spec`` free functions: ``Combiner.from_spec(spec)``
        returns a validated ``Combiner``, and rejects a spec whose type is not a ``Combiner``.
        """
        return _structure_component(spec, cls)

    @classmethod
    def build_spec(cls, spec: Mapping[str, Any], context: Any = None) -> Any:
        """``from_spec(spec).build(context)`` — replaces the per-family ``build_<family>`` free functions."""
        return _structure_component(spec, cls).build(context)


def register_live(cls: type) -> type:
    """Mark a live runtime type as non-portable: exporting a config that holds one raises loudly.

    The escape hatch — a config field typed ``Family | LiveFamily`` trains/builds with a live instance, but
    :func:`to_spec` on that config raises :class:`PortabilityError` instead of silently dropping it.
    """

    def _raise(_obj: Any) -> Any:
        raise PortabilityError(
            f"{cls.__name__} is a runtime-only instance and has no portable config; pass a Component spec "
            f"instead of a live object to make this config serializable."
        )

    _converter.register_unstructure_hook(cls, _raise)
    return cls


def _field_types(cls: type) -> dict[str, Any]:
    if cls not in _hints_cache:
        _hints_cache[cls] = get_type_hints(cls)
    return _hints_cache[cls]


def _is_component(tp: Any) -> bool:
    return isinstance(tp, type) and issubclass(tp, Component)


def _unstructure_field(value: Any) -> Any:
    if isinstance(value, Component):
        return to_spec(value)  # nested sub-component -> nested envelope
    return _converter.unstructure(value)  # leaves + live-instance guard (raises)


def to_spec(config: Component) -> dict[str, Any]:
    """config -> portable ``{type, version, config}`` dict (JSON-ready). Raises on live instances."""
    inner = {f.name: _unstructure_field(getattr(config, f.name)) for f in dataclasses.fields(config)}
    return {"type": config.__type_id__, "version": config.__version__, "config": inner}


def _structure_field(value: Any, ftype: Any) -> Any:
    args = get_args(ftype)
    if args:  # a Union (incl. Optional and the spec|live escape hatch)
        if value is None and type(None) in args:
            return None
        comp = next((a for a in args if _is_component(a)), None)
        if comp is not None and isinstance(value, Mapping) and "type" in value:
            return _structure_component(value, comp)  # always take the spec branch on load
        others = [a for a in args if a is not type(None) and not _is_component(a)]
        if len(others) == 1:
            return _structure_field(value, others[0])
        return _converter.structure(value, ftype)
    if _is_component(ftype):
        return _structure_component(value, ftype)
    return _converter.structure(value, ftype)  # cattrs does the leaf recursion + OmegaConf coercion


def _structure_component(spec: Any, expected: type[Component]) -> Component:
    if not isinstance(spec, Mapping):
        raise TypeError(f"Expected a spec mapping; found {type(spec).__name__}.")
    unknown_env = set(spec) - {"type", "version", "config"}
    if unknown_env:
        raise ValueError(f"Unknown spec field(s): {sorted(unknown_env)}.")
    missing = {"type", "version", "config"} - set(spec)
    if missing:
        raise ValueError(f"Missing spec field(s): {sorted(missing)}.")
    type_id, version, cfg = spec["type"], spec["version"], spec["config"]
    if not isinstance(type_id, str) or not type_id:
        raise TypeError("spec 'type' must be a non-empty string.")
    if not isinstance(version, int) or isinstance(version, bool) or version <= 0:
        raise TypeError("spec 'version' must be a positive integer.")
    if not isinstance(cfg, Mapping):
        raise TypeError("spec 'config' must be a mapping.")
    try:
        target = _REGISTRY[type_id]
    except KeyError:
        raise ValueError(f"Unknown type {type_id!r}; registered: {sorted(_REGISTRY)}.") from None
    if not issubclass(target, expected):
        raise ValueError(f"type {type_id!r} ({target.__name__}) is not a {expected.__name__}.")
    if version not in target.__versions__:
        raise ValueError(f"Unsupported {type_id!r} config version {version}; accepted: {sorted(target.__versions__)}.")
    field_types = _field_types(target)
    known = {f.name for f in dataclasses.fields(target)}
    unknown = set(cfg) - known
    if unknown:
        raise ValueError(f"Unknown field(s) for {type_id!r}: {sorted(unknown)}; allowed: {sorted(known)}.")
    kwargs = {name: _structure_field(cfg[name], field_types[name]) for name in cfg}
    return target(**kwargs)  # __post_init__ runs here (validation + canonicalization)


def parse(spec: Mapping[str, Any]) -> Component:
    """``{type, version, config}`` dict -> validated :class:`Component` (dispatched by the registry)."""
    return _structure_component(spec, Component)


def build(spec: Mapping[str, Any], context: Any = None) -> Any:
    """Parse then build in one call."""
    return _structure_component(spec, Component).build(context)
