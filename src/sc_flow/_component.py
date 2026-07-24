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
    type: str
    version: int
    config: dict[str, JsonValue]


class _Registration:
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
    def __init__(self, family: str) -> None:
        self._family = family
        self._registry: dict[str, _Registration] = {}

    @property
    def family(self) -> str:
        return self._family

    @property
    def types(self) -> list[str]:
        return sorted(self._registry)

    def register[C, R](
        self,
        type_id: str,
        *,
        config_type: type[C] | None = None,
        build: Callable[[C, ContextT], R] | None = None,
        versions: int | tuple[int, ...] = (1,),
    ) -> Callable[[type[C]], type[C]] | None:
        """Bind ``type_id`` to a config dataclass + its factory.

        Preferred form — decorate the config dataclass, which owns ``build(self, context) -> runtime`` so
        schema, ``__post_init__`` validation, and construction live in one class::

            @REGISTRY.register("pkg.thing")
            @dataclass(frozen=True)
            class ThingConfig:
                n: int

                def build(self, context):
                    return Thing(n=self.n, dim=context.dim)

        Low-level form — pass an explicit ``config_type`` + ``build`` factory (e.g. when the builder can't
        be a method on the config)::

            REGISTRY.register("pkg.thing", config_type=ThingConfig, build=_factory)

        Re-registering the same binding is idempotent (import-order friendly); a *different* binding for a
        known ``type_id`` is an error. ``versions`` is the accepted config-schema version set (a single int
        is promoted to a one-element set).
        """
        if config_type is None:
            # Decorator on the config dataclass: it exposes build(self, context); bind cls + cls.build.
            def _decorator(cls: type[C]) -> type[C]:
                factory = getattr(cls, "build", None)
                if not callable(factory):
                    raise TypeError(
                        f"{self._family} config {getattr(cls, '__name__', cls)!r} registered via "
                        f"@register('{type_id}') must define a 'build(self, context)' method."
                    )
                self.register(type_id, config_type=cls, build=factory, versions=versions)
                return cls

            return _decorator
        if build is None:
            raise TypeError(
                f"{self._family} register for {type_id!r} needs a 'build' factory when 'config_type' is given "
                f"(or use the decorator form on a config with a build() method)."
            )
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
            same = existing.config_type is config_type and existing.build is build and existing.versions == version_set
            if not same:
                raise ValueError(f"{self._family} component type {type_id!r} is already registered.")
            return
        self._registry[type_id] = registration

    def validate(self, spec: ComponentSpec | Mapping[str, Any]) -> ComponentSpec:
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
        validated = self.validate(spec)
        return self._parse(validated["type"], validated["version"], validated["config"])

    def build(self, spec: ComponentSpec | Mapping[str, Any], context: ContextT) -> Any:
        validated = self.validate(spec)
        config = self._parse(validated["type"], validated["version"], validated["config"])
        return self._registry[validated["type"]].build(config, context)

    def _parse(self, type_id: str, version: int, config: dict[str, JsonValue]) -> Any:
        try:
            registration = self._registry[type_id]
        except KeyError:
            raise ValueError(f"Unknown {self._family} type {type_id!r}; registered types are {self.types}.") from None
        if version not in registration.versions:
            raise ValueError(
                f"Unsupported {type_id!r} config version {version}; "
                f"supported versions: {sorted(registration.versions)}."
            )
        try:
            return registration.config_type(**config)
        except TypeError as e:
            raise ValueError(f"Invalid config for {self._family} type {type_id!r}: {e}") from e
