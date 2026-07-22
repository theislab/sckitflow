"""Generic discriminated-component specs, registries, and build contexts.

A runtime object (an :class:`torch.nn.Module`, an ``Objective``, ...) and its *persistent description* are
different things. The object holds weights and lives in memory; the description is immutable, JSON-safe data
that names an implementation and carries its parameters. This module provides the one small machinery both
the flow-matching toolbox and any sibling toolbox share for that description:

* :class:`ComponentSpec` — the wire object ``{type, version, config}``: a stable namespaced ``type``
  discriminator, a schema ``version`` for that type, and a JSON-only ``config`` payload. The *slot* a spec
  fills (``pooling``, ``combiner``, ...) determines its family, so the wire object never repeats it.
* :class:`ComponentRegistry` — a closed-or-open family of ``type -> (config dataclass, factory)`` bindings.
  It validates a spec without filling defaults or resolving aliases, parses the inner ``config`` into a
  frozen dataclass (whose ``__post_init__`` does the semantic validation), and builds the runtime object
  from a family-specific *build context* carrying the derived dimensions/device the caller never sizes by
  hand.

This is family-agnostic on purpose — it is part of the ``sc_flow.core`` (future ``mlcore``) surface, not the
flow-matching layer. See ``docs/plans/state.md`` §10 for the accepted contract this implements.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass
from typing import Any, TypedDict

__all__ = [
    "JsonValue",
    "ComponentSpec",
    "ComponentRegistry",
]

#: The value types JSON round-trips. A component ``config`` payload must contain only these.
type JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


class ComponentSpec(TypedDict):
    """The JSON wire representation of one polymorphic component.

    ``type`` is the stable, namespaced discriminator (e.g. ``"sc_flow.concat"`` — never a bare globally
    contended name like ``"concat"``); ``version`` versions *that type's* config schema; and ``config``
    contains JSON values only. Every field is required — there are no hidden defaults, and a spec cannot
    select an implementation through ``config`` alone. Runtime validation belongs to
    :meth:`ComponentRegistry.validate`.
    """

    type: str
    version: int
    config: dict[str, JsonValue]


class _Registration:
    """One ``type`` binding: its config dataclass, its supported versions, and its factory."""

    __slots__ = ("config_type", "build", "versions")

    def __init__(
        self,
        config_type: type[Any],
        build: Callable[..., Any],
        versions: frozenset[int],
    ) -> None:
        self.config_type = config_type
        self.build = build
        self.versions = versions


class ComponentRegistry[ContextT]:
    """A family of ``type -> (config dataclass, factory)`` bindings for one component slot.

    One registry serves one family (``"pooling"``, ``"combiner"``, ...). Each registered ``type`` binds a
    frozen config dataclass (its ``__post_init__`` validates semantics) to a factory
    ``build(config, context) -> runtime`` — where ``context`` is a family-specific
    :class:`BuildContext`-like object carrying derived dimensions/device. Validation is strict and
    default-free: a spec must name a registered ``type`` at a supported ``version`` and carry exactly the
    fields that ``type`` understands.

    The family may be *closed* (only built-ins register, as pooling and combiner are today) or later opened
    to third-party providers; the machinery is identical either way.
    """

    def __init__(self, family: str) -> None:
        self._family = family
        self._registry: dict[str, _Registration] = {}

    @property
    def family(self) -> str:
        """The slot name this registry serves (for error messages)."""
        return self._family

    @property
    def types(self) -> list[str]:
        """Sorted registered type ids."""
        return sorted(self._registry)

    def register[C, R](
        self,
        type_id: str,
        *,
        config_type: type[C],
        build: Callable[[C, ContextT], R],
        versions: int | tuple[int, ...] = (1,),
    ) -> None:
        """Bind ``type_id`` to a config dataclass and a factory.

        ``versions`` is the set of config-schema versions this binding accepts (a single int is promoted to
        a one-element set). Re-registering the same ``type_id`` to a different binding is an error; the same
        binding twice is idempotent (import-order friendly).
        """
        if not isinstance(type_id, str) or not type_id:
            raise TypeError(f"{self._family} component type must be a non-empty string.")
        if not (is_dataclass(config_type) and isinstance(config_type, type)):
            raise TypeError(f"{self._family} config_type for {type_id!r} must be a dataclass type.")
        version_set = frozenset((versions,) if isinstance(versions, int) else versions)
        if not version_set or any((not isinstance(v, int)) or isinstance(v, bool) or v <= 0 for v in version_set):
            raise ValueError(f"{self._family} versions for {type_id!r} must be positive integers.")
        existing = self._registry.get(type_id)
        registration = _Registration(config_type, build, version_set)
        if existing is not None:
            same = (
                existing.config_type is config_type
                and existing.build is build
                and existing.versions == version_set
            )
            if not same:
                raise ValueError(f"{self._family} component type {type_id!r} is already registered.")
            return
        self._registry[type_id] = registration

    def validate(self, spec: ComponentSpec | Mapping[str, Any]) -> ComponentSpec:
        """Validate an explicit spec without filling fields or resolving aliases.

        Checks the ``{type, version, config}`` envelope, that ``type`` is registered at a supported
        ``version``, that ``config`` holds JSON values only, and that it parses into the bound config
        dataclass. Returns a canonical spec whose ``config`` is the dataclass fields (so equal specs
        serialize identically). Raises :class:`ValueError`/:class:`TypeError` with an actionable message.
        """
        if not isinstance(spec, Mapping):
            raise TypeError(f"Expected a {self._family} spec mapping; found {type(spec).__name__}.")
        required = {"type", "version", "config"}
        unknown = set(spec) - required
        if unknown:
            raise ValueError(f"Unknown {self._family} spec field(s): {sorted(unknown)}.")
        missing = required - set(spec)
        if missing:
            raise ValueError(f"Missing {self._family} spec field(s): {sorted(missing)}.")
        type_id = spec["type"]
        version = spec["version"]
        raw_config = spec["config"]
        if not isinstance(type_id, str) or not type_id:
            raise TypeError(f"{self._family} spec 'type' must be a non-empty string.")
        if not isinstance(version, int) or isinstance(version, bool) or version <= 0:
            raise TypeError(f"{self._family} spec 'version' must be a positive integer.")
        if not isinstance(raw_config, Mapping):
            raise TypeError(f"{self._family} spec 'config' must be a mapping of JSON values.")
        config_dict = dict(raw_config)
        try:
            json.dumps(config_dict)
        except (TypeError, ValueError) as e:
            raise TypeError(f"{self._family} spec 'config' must contain JSON-serializable values only.") from e
        config = self._parse(type_id, version, config_dict)
        return ComponentSpec(type=type_id, version=version, config=asdict(config))

    def parse(self, spec: ComponentSpec | Mapping[str, Any]) -> Any:
        """Validate ``spec`` and return the parsed (frozen dataclass) config instance."""
        validated = self.validate(spec)
        return self._parse(validated["type"], validated["version"], validated["config"])

    def build(self, spec: ComponentSpec | Mapping[str, Any], context: ContextT) -> Any:
        """Build the runtime component from a validated spec and a family build context."""
        validated = self.validate(spec)
        config = self._parse(validated["type"], validated["version"], validated["config"])
        return self._registry[validated["type"]].build(config, context)

    def _parse(self, type_id: str, version: int, config: dict[str, JsonValue]) -> Any:
        try:
            registration = self._registry[type_id]
        except KeyError:
            raise ValueError(
                f"Unknown {self._family} type {type_id!r}; registered types are {self.types}."
            ) from None
        if version not in registration.versions:
            raise ValueError(
                f"Unsupported {type_id!r} config version {version}; "
                f"supported versions: {sorted(registration.versions)}."
            )
        try:
            return registration.config_type(**config)
        except TypeError as e:
            raise ValueError(f"Invalid config for {self._family} type {type_id!r}: {e}") from e
