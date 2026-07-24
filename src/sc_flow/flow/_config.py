
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import torch
from sc_flow._component import JsonValue
from sc_flow.nn._modules import FunctionalModule
from sc_flow.nn._utils import init_module_from_dict

from sc_flow._types import TimeFeaturesId
from sc_flow.flow._combiner import BaseCombiner, CombinerSpec, build_combiner, validate_combiner_spec
from sc_flow.flow._pooling import BasePooling, PoolingSpec, build_pooling, validate_pooling_spec
from sc_flow.flow._realm_encoders import (
    RealmEncoderSpec,
    build_realm_encoder,
    realm_output_dim,
    validate_realm_encoder_spec,
)
from sc_flow.flow._time_features import get_time_features_fn

if TYPE_CHECKING:
    from sc_flow.flow._set_encoder import SetEncoder
    from sc_flow.flow._vf import MLPVelocity

__all__ = [
    "FORMAT_VERSION",
    "ARCHITECTURE_TYPE",
    "MLPEmbedderConfig",
    "SetEncoderConfig",
    "SetEncoderContext",
    "MLPVelocityConfig",
    "VelocityFieldContext",
]

#: Bump when the *bundle envelope* (``config.json`` layout) changes incompatibly.
FORMAT_VERSION = 1
#: Stable discriminator for the one built-in top-level architecture.
ARCHITECTURE_TYPE = "sc_flow.mlp_velocity"
#: Config-schema version for :data:`ARCHITECTURE_TYPE`.
ARCHITECTURE_VERSION = 1


def _ensure_json(value: Any, where: str) -> Any:
    try:
        json.dumps(value)
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"{where} is not JSON-serializable ({e}). Portable configs may only contain JSON values; "
            f"pass parameter-free choices (e.g. activations) as string ids rather than classes."
        ) from e
    return value


@dataclass
class MLPEmbedderConfig:

    output_dim: int
    mlp_kwargs: dict[str, Any] = field(default_factory=dict)

    def build(self, input_dim: int) -> torch.nn.Module:
        """Build the stream embedder MLP over ``input_dim -> output_dim`` (config owns its construction).

        A presence-based *stream* embedder: the caller passes ``None`` (no embedder, raw stream) or this
        config, which builds the MLP. ``input_dim`` is supplied by the velocity field at build time; an
        ``input_dim`` carried in ``mlp_kwargs`` (e.g. the GENOT source embedder) is overridden by it.
        """
        return init_module_from_dict(dict(self.mlp_kwargs), input_dim=input_dim, output_dim=self.output_dim)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "output_dim": self.output_dim,
            "mlp_kwargs": _ensure_json(dict(self.mlp_kwargs), "MLPEmbedderConfig.mlp_kwargs"),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MLPEmbedderConfig:
        unknown = set(data) - {"output_dim", "mlp_kwargs"}
        if unknown:
            raise ValueError(f"Unknown MLPEmbedderConfig field(s): {sorted(unknown)}.")
        if "output_dim" not in data:
            raise ValueError("MLPEmbedderConfig requires an explicit 'output_dim'.")
        return cls(output_dim=data["output_dim"], mlp_kwargs=dict(data.get("mlp_kwargs", {})))


def _embedder_to_dict(embedder: MLPEmbedderConfig | None) -> dict[str, JsonValue] | None:
    return None if embedder is None else embedder.to_dict()


def _embedder_from_dict(data: dict[str, Any] | None) -> MLPEmbedderConfig | None:
    return None if data is None else MLPEmbedderConfig.from_dict(data)


def _build_embedder(config: MLPEmbedderConfig | None, input_dim: int) -> torch.nn.Module:
    """A present :class:`MLPEmbedderConfig` builds its MLP; ``None`` passes the raw stream through."""
    return torch.nn.Identity() if config is None else config.build(input_dim)


@dataclass(frozen=True)
class SetEncoderContext:
    """Build context for a :class:`SetEncoderConfig`.

    Empty today — a :class:`SetEncoderConfig` is dim-complete (realm specs carry their own vocab/feature
    widths, ``output_dim`` is explicit, ``pooling_proj_dim`` is resolved from the pooled realm widths).
    Kept as a named type for parity with the pooling/combiner/realm build contexts and as the extension
    point for future derived input (e.g. device placement).
    """


@dataclass
class SetEncoderConfig:

    realms: dict[str, RealmEncoderSpec]  # realm -> its encoder spec (embedding / onehot / feature_mlp / ...)
    output_dim: int
    #: A portable :class:`PoolingSpec` (JSON) or a runtime-only :class:`BasePooling` instance (the research
    #: escape hatch — trains normally, but ``to_dict``/``to_config`` raise because it has no portable form).
    pooling: PoolingSpec | BasePooling
    pooling_proj_dim: int | None = None
    pooling_proj_bias: bool = True
    covariates_not_pooled: list[str] = field(default_factory=list)
    output_layers_kwargs: dict[str, Any] = field(default_factory=dict)
    condition_mode: str = "deterministic"

    def __post_init__(self) -> None:
        # Canonicalize the leaf specs so equal encoders serialize identically; a custom BasePooling instance
        # is left as-is (it is a runtime object, not a spec).
        if not isinstance(self.pooling, BasePooling):
            self.pooling = validate_pooling_spec(self.pooling)
        self.realms = {realm: validate_realm_encoder_spec(spec) for realm, spec in self.realms.items()}
        self.covariates_not_pooled = list(self.covariates_not_pooled)
        if self.condition_mode not in ("deterministic", "stochastic"):
            raise ValueError(
                f"condition_mode must be 'deterministic' or 'stochastic', found {self.condition_mode!r}."
            )
        from sc_flow._utils import check_sequence_query_against_reference

        check_sequence_query_against_reference(
            self.covariates_not_pooled,
            self.realms.keys(),
            allow_missing_from_query=True,
            allow_missing_from_reference=False,
        )

    # --- derived sizing (pure functions of the config; the model never sizes anything by hand) ---------

    @property
    def is_custom_pooling(self) -> bool:
        return isinstance(self.pooling, BasePooling)

    @property
    def is_stochastic(self) -> bool:
        return self.condition_mode == "stochastic"

    @property
    def covariates_pooled(self) -> list[str]:
        return [cov for cov in self.realms if cov not in self.covariates_not_pooled]

    @property
    def _min_pooled_dims(self) -> int | None:
        pooled = self.covariates_pooled
        if not pooled:
            return None
        return min(realm_output_dim(self.realms[cov]) for cov in pooled)

    @property
    def resolved_pooling_proj_dim(self) -> int | None:
        return self.pooling_proj_dim if self.pooling_proj_dim is not None else self._min_pooled_dims

    @property
    def pooling_output_dim(self) -> int:
        if not self.covariates_pooled:
            return 0
        if isinstance(self.pooling, BasePooling):
            return self.pooling.output_dim
        if self.pooling["type"] == "sc_flow.attention_seed":
            return int(self.pooling["config"]["v_dim"])
        return int(self.resolved_pooling_proj_dim)

    @property
    def decoder_input_dim(self) -> int:
        not_pooled_dims = [
            realm_output_dim(self.realms[cov]) for cov in self.realms if cov in self.covariates_not_pooled
        ]
        return self.pooling_output_dim + sum(not_pooled_dims)

    @property
    def pooling_spec(self) -> PoolingSpec | None:
        if isinstance(self.pooling, BasePooling):
            return None
        return PoolingSpec(
            type=self.pooling["type"], version=self.pooling["version"], config=dict(self.pooling["config"])
        )

    # --- construction (config owns build) --------------------------------------------------------------

    def build(self, context: SetEncoderContext | None = None) -> SetEncoder:
        from sc_flow.flow._set_encoder import SetEncoder

        proj_dim = self.resolved_pooling_proj_dim
        if self.covariates_pooled and (proj_dim is None or proj_dim <= 0):
            raise ValueError(f"pooling_proj_dim must be positive when covariates are pooled, found {proj_dim}.")

        if self.is_custom_pooling:
            if not self.covariates_pooled:
                raise ValueError(
                    "A custom pooling instance was supplied, but no covariates are configured for pooling."
                )
            if self.pooling.input_dim != proj_dim:
                raise ValueError(
                    f"Custom pooling expects input_dim={self.pooling.input_dim}, but SetEncoder projects to "
                    f"pooling_proj_dim={proj_dim}."
                )

        input_layers = torch.nn.ModuleDict(
            {realm: build_realm_encoder(spec) for realm, spec in self.realms.items()}
        )
        proj_layers = torch.nn.ModuleDict(
            {
                cov: torch.nn.Linear(realm_output_dim(self.realms[cov]), proj_dim, bias=self.pooling_proj_bias)
                for cov in self.covariates_pooled
            }
        )
        if not self.covariates_pooled:
            pooling_layer: torch.nn.Module = torch.nn.Identity()
        elif self.is_custom_pooling:
            pooling_layer = self.pooling
        else:
            pooling_layer = build_pooling(self.pooling, input_dim=proj_dim)

        layers: dict[str, torch.nn.Module] = {
            "input_layers": input_layers,
            "proj_layers": proj_layers,
            "pooling_layer": pooling_layer,
            "output_layer": init_module_from_dict(
                dict(self.output_layers_kwargs), input_dim=self.decoder_input_dim, output_dim=self.output_dim
            ),
        }
        # A stochastic (VAE-style) encoder gets a second head for the log-variance (same shape as the mean).
        if self.is_stochastic:
            layers["var_layer"] = init_module_from_dict(
                dict(self.output_layers_kwargs), input_dim=self.decoder_input_dim, output_dim=self.output_dim
            )

        # Bake the resolved projection width back into the stored config so ``to_config`` round-trips the
        # exact number the module was built with.
        resolved = replace(self, pooling_proj_dim=proj_dim)
        return SetEncoder(resolved, torch.nn.ModuleDict(layers))

    def to_dict(self) -> dict[str, JsonValue]:
        if self.is_custom_pooling:
            raise ValueError(
                "This SetEncoder uses a custom BasePooling instance (runtime-only) and has no portable "
                "config. Build it with a PoolingSpec to enable export."
            )
        return {
            "realms": {realm: dict(spec) for realm, spec in self.realms.items()},
            "output_dim": self.output_dim,
            "pooling": dict(self.pooling),
            "pooling_proj_dim": self.pooling_proj_dim,
            "pooling_proj_bias": self.pooling_proj_bias,
            "covariates_not_pooled": list(self.covariates_not_pooled),
            "output_layers_kwargs": _ensure_json(
                dict(self.output_layers_kwargs), "SetEncoderConfig.output_layers_kwargs"
            ),
            "condition_mode": self.condition_mode,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SetEncoderConfig:
        allowed = {
            "realms",
            "output_dim",
            "pooling",
            "pooling_proj_dim",
            "pooling_proj_bias",
            "covariates_not_pooled",
            "output_layers_kwargs",
            "condition_mode",
        }
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"Unknown SetEncoderConfig field(s): {sorted(unknown)}.")
        missing = {"realms", "output_dim", "pooling"} - set(data)
        if missing:
            raise ValueError(f"Missing required SetEncoderConfig field(s): {sorted(missing)}.")
        return cls(
            realms={realm: dict(spec) for realm, spec in data["realms"].items()},
            output_dim=data["output_dim"],
            pooling=data["pooling"],
            pooling_proj_dim=data.get("pooling_proj_dim"),
            pooling_proj_bias=data.get("pooling_proj_bias", True),
            covariates_not_pooled=list(data.get("covariates_not_pooled", [])),
            output_layers_kwargs=dict(data.get("output_layers_kwargs", {})),
            condition_mode=data.get("condition_mode", "deterministic"),
        )


@dataclass(frozen=True)
class VelocityFieldContext:
    """Build context for a :class:`MLPVelocityConfig`.

    Empty today — a dim-complete :class:`MLPVelocityConfig` sizes every child from its own fields
    (``state_dim`` + embedder widths + the condition/source encoder widths). Kept as a named type for
    parity with the nested build contexts and the extension point for future derived input.
    """


@dataclass
class MLPVelocityConfig:

    state_dim: int
    #: A portable :class:`CombinerSpec` (JSON) or a runtime-only :class:`BaseCombiner` instance (the research
    #: escape hatch — trains normally, but ``to_dict``/``to_config`` raise because it has no portable form).
    combiner: CombinerSpec | BaseCombiner
    state_embedder: MLPEmbedderConfig | None = None
    time_embedder: MLPEmbedderConfig | None = None
    source_embedder: MLPEmbedderConfig | None = None
    time_features_id: TimeFeaturesId | None = None
    num_time_features: int | None = None
    max_period: int | None = None
    vf_decoder_mlp_kwargs: dict[str, Any] = field(default_factory=dict)
    condition_encoder: SetEncoderConfig | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.combiner, BaseCombiner):
            self.combiner = validate_combiner_spec(self.combiner)

    # --- derived sizing (pure functions of the config) -------------------------------------------------

    @property
    def is_custom_combiner(self) -> bool:
        return isinstance(self.combiner, BaseCombiner)

    @property
    def is_conditional(self) -> bool:
        return self.condition_encoder is not None

    @property
    def use_source_embedder(self) -> bool:
        return self.source_embedder is not None

    @staticmethod
    def _embed_output_dim(config: MLPEmbedderConfig | None, raw_dim: int) -> int:
        return raw_dim if config is None else config.output_dim

    @property
    def num_time_features_resolved(self) -> int:
        if self.time_features_id is None:
            return 1
        if self.num_time_features is None:
            raise ValueError(
                "num_time_features is required when time_features_id is set "
                f"(got time_features_id={self.time_features_id!r}, num_time_features=None)."
            )
        return self.num_time_features

    @property
    def source_embedder_input_dim(self) -> int:
        carried = self.source_embedder.mlp_kwargs.get("input_dim") if self.source_embedder is not None else None
        return self.state_dim if carried is None else int(carried)

    @property
    def _source_embedder_output_dim(self) -> int:
        return self._embed_output_dim(self.source_embedder, self.source_embedder_input_dim)

    def _condition_dim(self, condition_output_dim: int | None) -> int | None:
        src = self._source_embedder_output_dim if self.use_source_embedder else None
        if condition_output_dim is not None and src is not None:
            return condition_output_dim + src
        return condition_output_dim if condition_output_dim is not None else src

    # --- construction (config owns build) --------------------------------------------------------------

    def build(self, context: VelocityFieldContext | None = None) -> MLPVelocity:
        from sc_flow.flow._vf import MLPVelocity

        num_time_features = self.num_time_features_resolved

        # Build the condition (set) encoder first: it fixes the conditioning width the combiner consumes.
        condition_encoder = None if self.condition_encoder is None else self.condition_encoder.build()
        condition_output_dim = None if condition_encoder is None else condition_encoder.output_dim

        latent_state_dim = self._embed_output_dim(self.state_embedder, self.state_dim)
        latent_time_dim = self._embed_output_dim(self.time_embedder, num_time_features)
        latent_condition_dim = self._condition_dim(condition_output_dim)

        if self.is_custom_combiner:
            got = (
                self.combiner._latent_state_dim,
                self.combiner._latent_time_dim,
                self.combiner._latent_condition_dim,
            )
            want = (latent_state_dim, latent_time_dim, latent_condition_dim)
            if got != want:
                raise ValueError(
                    f"Custom combiner was sized for (state, time, condition)={got}, but this velocity "
                    f"field feeds it {want}."
                )
            combiner = self.combiner
        else:
            combiner = build_combiner(
                self.combiner,
                latent_state_dim=latent_state_dim,
                latent_time_dim=latent_time_dim,
                latent_condition_dim=latent_condition_dim,
            )

        modules: dict[str, torch.nn.Module] = {
            "time_features": FunctionalModule(
                get_time_features_fn(
                    num_time_features=num_time_features,
                    time_features_id=self.time_features_id,
                    max_period=self.max_period,
                )
            ),
            "time_embedder": _build_embedder(self.time_embedder, num_time_features),
            "state_embedder": _build_embedder(self.state_embedder, self.state_dim),
            "combiner": combiner,
        }
        if condition_encoder is not None:
            modules["condition_encoder"] = condition_encoder
        if self.use_source_embedder:
            modules["source_embedder"] = _build_embedder(self.source_embedder, self.source_embedder_input_dim)
        modules["vf_decoder"] = init_module_from_dict(
            dict(self.vf_decoder_mlp_kwargs), input_dim=combiner.output_dim, output_dim=self.state_dim
        )

        return MLPVelocity(self, torch.nn.ModuleDict(modules))

    def to_dict(self) -> dict[str, JsonValue]:
        if self.is_custom_combiner:
            raise ValueError(
                "This velocity field uses a custom BaseCombiner instance (runtime-only) and has no portable "
                "config. Build it with a CombinerSpec to enable export."
            )
        return {
            "state_dim": self.state_dim,
            "combiner": dict(self.combiner),
            "state_embedder": _embedder_to_dict(self.state_embedder),
            "time_embedder": _embedder_to_dict(self.time_embedder),
            "source_embedder": _embedder_to_dict(self.source_embedder),
            "time_features_id": self.time_features_id,
            "num_time_features": self.num_time_features,
            "max_period": self.max_period,
            "vf_decoder_mlp_kwargs": _ensure_json(
                dict(self.vf_decoder_mlp_kwargs), "MLPVelocityConfig.vf_decoder_mlp_kwargs"
            ),
            "condition_encoder": None if self.condition_encoder is None else self.condition_encoder.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MLPVelocityConfig:
        allowed = {
            "state_dim",
            "combiner",
            "state_embedder",
            "time_embedder",
            "source_embedder",
            "time_features_id",
            "num_time_features",
            "max_period",
            "vf_decoder_mlp_kwargs",
            "condition_encoder",
        }
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"Unknown MLPVelocityConfig field(s): {sorted(unknown)}.")
        missing = {"state_dim", "combiner"} - set(data)
        if missing:
            raise ValueError(f"Missing required MLPVelocityConfig field(s): {sorted(missing)}.")
        condition_encoder = data.get("condition_encoder")
        return cls(
            state_dim=data["state_dim"],
            combiner=data["combiner"],
            state_embedder=_embedder_from_dict(data.get("state_embedder")),
            time_embedder=_embedder_from_dict(data.get("time_embedder")),
            source_embedder=_embedder_from_dict(data.get("source_embedder")),
            time_features_id=data.get("time_features_id"),
            num_time_features=data.get("num_time_features"),
            max_period=data.get("max_period"),
            vf_decoder_mlp_kwargs=dict(data.get("vf_decoder_mlp_kwargs", {})),
            condition_encoder=None if condition_encoder is None else SetEncoderConfig.from_dict(condition_encoder),
        )

    def to_spec(self) -> dict[str, JsonValue]:
        return {"type": ARCHITECTURE_TYPE, "version": ARCHITECTURE_VERSION, "config": self.to_dict()}

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> MLPVelocityConfig:
        if spec.get("type") != ARCHITECTURE_TYPE:
            raise ValueError(f"Expected architecture type {ARCHITECTURE_TYPE!r}, found {spec.get('type')!r}.")
        if spec.get("version") != ARCHITECTURE_VERSION:
            raise ValueError(
                f"Unsupported {ARCHITECTURE_TYPE!r} config version {spec.get('version')!r}; "
                f"supported: [{ARCHITECTURE_VERSION}]."
            )
        return cls.from_dict(spec["config"])
