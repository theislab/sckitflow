
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from sc_flow._types import TimeFeaturesId
from scfit._component import JsonValue
from sc_flow.flow._combiner import CombinerSpec, validate_combiner_spec
from sc_flow.flow._pooling import PoolingSpec, validate_pooling_spec

__all__ = [
    "FORMAT_VERSION",
    "ARCHITECTURE_TYPE",
    "MLPEmbedderConfig",
    "SetEncoderConfig",
    "MLPVelocityConfig",
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


@dataclass
class SetEncoderConfig:

    input_layers: dict[str, dict[str, Any]]
    output_dim: int
    pooling: PoolingSpec
    pooling_proj_dim: int | None = None
    pooling_proj_bias: bool = True
    covariates_not_pooled: list[str] = field(default_factory=list)
    output_layers_kwargs: dict[str, Any] = field(default_factory=dict)
    condition_mode: str = "deterministic"

    def __post_init__(self) -> None:
        # Canonicalize the leaf pooling spec so equal encoders serialize identically.
        self.pooling = validate_pooling_spec(self.pooling)
        if self.condition_mode not in ("deterministic", "stochastic"):
            raise ValueError(
                f"condition_mode must be 'deterministic' or 'stochastic', found {self.condition_mode!r}."
            )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "input_layers": _ensure_json(
                {k: dict(v) for k, v in self.input_layers.items()}, "SetEncoderConfig.input_layers"
            ),
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
            "input_layers",
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
        missing = {"input_layers", "output_dim", "pooling"} - set(data)
        if missing:
            raise ValueError(f"Missing required SetEncoderConfig field(s): {sorted(missing)}.")
        return cls(
            input_layers={k: dict(v) for k, v in data["input_layers"].items()},
            output_dim=data["output_dim"],
            pooling=data["pooling"],
            pooling_proj_dim=data.get("pooling_proj_dim"),
            pooling_proj_bias=data.get("pooling_proj_bias", True),
            covariates_not_pooled=list(data.get("covariates_not_pooled", [])),
            output_layers_kwargs=dict(data.get("output_layers_kwargs", {})),
            condition_mode=data.get("condition_mode", "deterministic"),
        )


@dataclass
class MLPVelocityConfig:

    state_dim: int
    combiner: CombinerSpec
    state_embedder: MLPEmbedderConfig | None = None
    time_embedder: MLPEmbedderConfig | None = None
    source_embedder: MLPEmbedderConfig | None = None
    time_features_id: TimeFeaturesId | None = None
    num_time_features: int | None = None
    max_period: int | None = None
    vf_decoder_mlp_kwargs: dict[str, Any] = field(default_factory=dict)
    condition_encoder: SetEncoderConfig | None = None

    def __post_init__(self) -> None:
        self.combiner = validate_combiner_spec(self.combiner)

    def to_dict(self) -> dict[str, JsonValue]:
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
