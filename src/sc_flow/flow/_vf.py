from __future__ import annotations

import abc
from collections.abc import Callable

import torch

from scfit._types import LayersDict
from sc_flow._types import TimeFeaturesId
from sc_flow.flow._torch_types import MappedTensor
from sc_flow.flow._torch_utils import make_concatenation_possible
from scfit.nn._modules import BaseModule, FunctionalModule
from scfit.nn._utils import init_module_from_dict
from sc_flow.flow._combiner import BaseCombiner, CombinerSpec, build_combiner, validate_combiner_spec
from sc_flow.flow._config import MLPEmbedderConfig, MLPVelocityConfig
from sc_flow.flow._set_encoder import SetEncoder
from sc_flow.flow._time_features import get_time_features_fn

__all__ = [
    "BaseVelocityField",
    "MLPEmbedderConfig",
    "MLPVelocity",
    "VelocityFieldFn",
]

#: Callable form of a velocity field, ``(t, x) -> velocity``, as fed to ODE/SDE solvers. Lives here (the
#: flow-matching layer) rather than in the generic core: it is a flow-matching concept.
VelocityFieldFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class BaseVelocityField(BaseModule):

    @abc.abstractmethod
    def forward(
        self,
        t: torch.Tensor,
        xt: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def get_vf_fn(
        self,
        *args,
        **kwargs,
    ) -> VelocityFieldFn:
        ...


class MLPVelocity(BaseVelocityField):

    def __init__(
        self,
        state_dim: int,
        state_embedder: MLPEmbedderConfig | None = None,
        time_embedder: MLPEmbedderConfig | None = None,
        source_embedder: MLPEmbedderConfig | None = None,
        time_features_id: TimeFeaturesId | None = None,
        num_time_features: int | None = None,
        max_period: int | None = None,
        vf_decoder_mlp_kwargs: LayersDict | None = None,
        combiner: CombinerSpec | BaseCombiner | None = None,
        condition_encoder: SetEncoder | None = None,
    ) -> None:
        super().__init__()
        self._state_dim = state_dim
        # Each stream embedder is presence-based: an MLPEmbedderConfig builds an MLP over that stream (with a
        # required, explicit output width), None passes the raw stream through.
        self._state_embedder = state_embedder
        self._time_embedder = time_embedder
        self._source_embedder = source_embedder
        self._time_features_id = time_features_id
        self._num_time_features = num_time_features
        self._max_period = max_period
        self._vf_decoder_mlp_kwargs = {} if vf_decoder_mlp_kwargs is None else vf_decoder_mlp_kwargs
        # A CombinerSpec (mapping) is validated/canonicalized now and saved/restored via config.json; a
        # custom BaseCombiner instance is not saved (see BaseModule) and must be passed again to
        # from_pretrained — otherwise the strict weight load raises a clear mismatch error. None is rejected
        # at build time (no hidden default combiner).
        self._combiner = combiner if (combiner is None or isinstance(combiner, BaseCombiner)) else (
            validate_combiner_spec(combiner)
        )
        self._condition_encoder = condition_encoder

        self._vf = self._make_modules()

    @property
    def _use_time_features(
        self,
    ) -> bool:
        return self._time_features_id is not None

    @property
    def use_source_embedder(
        self,
    ) -> bool:
        return self._source_embedder is not None

    @staticmethod
    def _embed_output_dim(config: MLPEmbedderConfig | None, raw_dim: int) -> int:
        return raw_dim if config is None else config.output_dim

    @property
    def _source_embedder_input_dim(self) -> int:
        input_dim = self._source_embedder.mlp_kwargs.get("input_dim") if self._source_embedder is not None else None
        return self._state_dim if input_dim is None else input_dim

    @property
    def _source_embedder_output_dim(self) -> int:
        return self._embed_output_dim(self._source_embedder, self._source_embedder_input_dim)

    @property
    def _combiner_dim(
        self,
    ) -> int | None:
        cond = self._condition_encoder.output_dim if self.is_conditional else None
        src = self._source_embedder_output_dim if self.use_source_embedder else None
        if cond is not None and src is not None:
            return cond + src
        return cond if cond is not None else src

    def _get_num_time_features(
        self,
    ) -> int:
        if not self._use_time_features:
            return 1
        if self._num_time_features is None:
            raise ValueError(
                "num_time_features is required when time_features_id is set "
                f"(got time_features_id={self._time_features_id!r}, num_time_features=None)."
            )
        return self._num_time_features

    def condition_stats(
        self,
        condition_dict: MappedTensor | None = None,
        condition_mask: MappedTensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.is_conditional:
            return None, None
        if condition_dict is None:
            msg = "Conditional VFs should take a condition as input, found `None`."
            raise TypeError(msg)
        return self._vf["condition_encoder"](condition_dict, condition_mask=condition_mask)

    def _get_encoded_source(
        self,
        source: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.use_source_embedder:
            if source is None:
                msg = "When using a source embedder a source state should be passed, found `None`."
                raise TypeError(msg)
            return self._vf["source_embedder"](source)
        return None

    def _get_combiner_input(
        self, encoded_condition: torch.Tensor, encoded_source: torch.tensor
    ) -> torch.Tensor | None:
        if encoded_condition is not None and encoded_source is not None:
            return torch.concatenate(
                (make_concatenation_possible(encoded_condition, encoded_source, -1), encoded_source), dim=-1
            )
        elif encoded_condition is None and encoded_source is not None:
            return encoded_source
        elif encoded_condition is not None and encoded_source is None:
            return encoded_condition
        return None

    def _make_time_features(
        self,
    ) -> FunctionalModule:
        time_features_fn = get_time_features_fn(
            num_time_features=self._get_num_time_features(),
            time_features_id=self._time_features_id,
            max_period=self._max_period,
        )
        return FunctionalModule(time_features_fn)

    def _make_embedder(
        self,
        config: MLPEmbedderConfig | None,
        input_dim: int,
    ) -> BaseModule | torch.nn.Identity:
        if config is None:
            return torch.nn.Identity()
        return init_module_from_dict(dict(config.mlp_kwargs), input_dim=input_dim, output_dim=config.output_dim)

    @property
    def _combiner_dims(self) -> tuple[int, int, int | None]:
        return (
            self._embed_output_dim(self._state_embedder, self._state_dim),
            self._embed_output_dim(self._time_embedder, self._get_num_time_features()),
            self._combiner_dim,
        )

    def _make_combiner(
        self,
    ) -> BaseCombiner:
        latent_state_dim, latent_time_dim, latent_condition_dim = self._combiner_dims
        if isinstance(self._combiner, BaseCombiner):
            got = (
                self._combiner._latent_state_dim,
                self._combiner._latent_time_dim,
                self._combiner._latent_condition_dim,
            )
            if got != (latent_state_dim, latent_time_dim, latent_condition_dim):
                msg = (
                    f"Custom combiner was sized for (state, time, condition)={got}, but this velocity "
                    f"field feeds it {(latent_state_dim, latent_time_dim, latent_condition_dim)}."
                )
                raise ValueError(msg)
            return self._combiner
        if self._combiner is None:
            raise ValueError(
                "A combiner must be provided explicitly: pass a CombinerSpec (e.g. "
                '{"type": "sc_flow.concat", "version": 1, "config": {}}) or a BaseCombiner instance. '
                "There is no default combiner."
            )
        return build_combiner(
            self._combiner,
            latent_state_dim=latent_state_dim,
            latent_time_dim=latent_time_dim,
            latent_condition_dim=latent_condition_dim,
        )

    def _make_vf_decoder(
        self,
        decoder_input_dim: int,
    ) -> BaseModule:
        return init_module_from_dict(
            self._vf_decoder_mlp_kwargs,
            input_dim=decoder_input_dim,
            output_dim=self._state_dim,
        )

    def _make_modules(
        self,
    ) -> torch.nn.Module:
        modules = {
            "time_features": self._make_time_features(),
            "time_embedder": self._make_embedder(self._time_embedder, self._get_num_time_features()),
            "state_embedder": self._make_embedder(self._state_embedder, self._state_dim),
            "combiner": self._make_combiner(),
        }
        if self.is_conditional:
            modules["condition_encoder"] = self._condition_encoder
        if self.use_source_embedder:
            modules["source_embedder"] = self._make_embedder(
                self._source_embedder, self._source_embedder_input_dim
            )
        modules["vf_decoder"] = self._make_vf_decoder(
            modules["combiner"].output_dim,
        )
        return torch.nn.ModuleDict(modules)

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        condition_dict: MappedTensor | None = None,
        source: torch.Tensor | None = None,
        condition_mask: MappedTensor | None = None,
    ) -> torch.Tensor:
        # Inference/solver path: use the condition **mean** embedding (a stochastic encoder's inference
        # is its mean, i.e. encoder_noise = 0). Training reparameterizes upstream and calls
        # :meth:`velocity_from_embedding` directly.
        mean, _ = self.condition_stats(condition_dict, condition_mask=condition_mask)
        return self.velocity_from_embedding(t, x, mean, source=source)

    def velocity_from_embedding(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        cond_embedding: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoded_xt = self._vf["state_embedder"](x)
        encoded_t = self._vf["time_embedder"](self._vf["time_features"](t))
        encoded_source = self._get_encoded_source(source)
        encoded_concat = self._vf["combiner"](
            encoded_t, encoded_xt, encoded_condition=self._get_combiner_input(cond_embedding, encoded_source)
        )
        return self._vf["vf_decoder"](encoded_concat)

    def get_vf_fn(
        self,
        condition_dict: MappedTensor | None = None,
        source: torch.Tensor | None = None,
        condition_mask: MappedTensor | None = None,
    ) -> VelocityFieldFn:

        def _vf_fn(t: torch.Tensor, x: torch.Tensor):
            return self.forward(
                t,
                x,
                condition_dict=condition_dict,
                source=source,
                condition_mask=condition_mask,
            )

        return _vf_fn

    @property
    def is_conditional(
        self,
    ) -> bool:
        return self._condition_encoder is not None

    @property
    def is_stochastic(
        self,
    ) -> bool:
        return self.is_conditional and self._condition_encoder.is_stochastic

    def to_config(self) -> MLPVelocityConfig:
        if self._combiner is None:
            raise ValueError("Velocity field has no combiner; nothing to serialize.")
        if isinstance(self._combiner, BaseCombiner):
            raise ValueError(
                "This velocity field uses a custom BaseCombiner instance (runtime-only) and has no portable "
                "config. Build it with a CombinerSpec to enable export."
            )
        condition_encoder = None if self._condition_encoder is None else self._condition_encoder.to_config()
        return MLPVelocityConfig(
            state_dim=self._state_dim,
            combiner=dict(self._combiner),
            state_embedder=self._state_embedder,
            time_embedder=self._time_embedder,
            source_embedder=self._source_embedder,
            time_features_id=self._time_features_id,
            num_time_features=self._num_time_features,
            max_period=self._max_period,
            vf_decoder_mlp_kwargs=dict(self._vf_decoder_mlp_kwargs),
            condition_encoder=condition_encoder,
        )

    @classmethod
    def from_config(cls, config: MLPVelocityConfig) -> MLPVelocity:
        condition_encoder = (
            None if config.condition_encoder is None else SetEncoder.from_config(config.condition_encoder)
        )
        return cls(
            state_dim=config.state_dim,
            state_embedder=config.state_embedder,
            time_embedder=config.time_embedder,
            source_embedder=config.source_embedder,
            time_features_id=config.time_features_id,
            num_time_features=config.num_time_features,
            max_period=config.max_period,
            vf_decoder_mlp_kwargs=dict(config.vf_decoder_mlp_kwargs),
            combiner=dict(config.combiner),
            condition_encoder=condition_encoder,
        )

    def save_pretrained(self, save_directory: str, **kwargs) -> None:
        import json
        from pathlib import Path

        from safetensors.torch import save_model

        spec = self.to_config().to_spec()
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(json.dumps(spec, indent=2))
        # save_model (not save_file): the condition encoder is registered under two names, so its tensors
        # are shared in the state_dict; save_model dedupes them, load_model restores both references.
        save_model(self, str(path / "model.safetensors"))

    @classmethod
    def from_pretrained(cls, save_directory: str, **kwargs) -> MLPVelocity:
        import json
        from pathlib import Path

        from safetensors.torch import load_model

        path = Path(save_directory)
        spec = json.loads((path / "config.json").read_text())
        model = cls.from_config(MLPVelocityConfig.from_spec(spec))
        load_model(model, str(path / "model.safetensors"), strict=True)
        return model
