from __future__ import annotations

import torch

from sc_flow.flow._torch_types import MappedTensor
from sc_flow.flow._torch_utils import make_concatenation_possible
from sc_flow.flow._config import MLPEmbedderConfig, MLPVelocityConfig, VelocityFieldContext

__all__ = [
    "MLPEmbedderConfig",
    "MLPVelocity",
]


class MLPVelocity(torch.nn.Module):
    """MLP velocity field ``v(t, x_t, cond)``. Dumb by construction: :meth:`MLPVelocityConfig.build` sizes
    and assembles the ``ModuleDict`` (recursively building the combiner / condition encoder from their own
    configs) and hands it here. This module only runs ``forward`` and reports its config via ``to_config``.
    """

    def __init__(self, config: MLPVelocityConfig, modules: torch.nn.ModuleDict) -> None:
        super().__init__()
        self._config = config
        self._vf = modules

    @property
    def is_conditional(self) -> bool:
        return self._config.is_conditional

    @property
    def is_stochastic(self) -> bool:
        return self.is_conditional and self._vf["condition_encoder"].is_stochastic

    @property
    def use_source_embedder(self) -> bool:
        return self._config.use_source_embedder

    # --- classifier-free guidance (CFG) --------------------------------------------------------------

    @property
    def condition_dropout_prob(self) -> float:
        return float(self._config.condition_dropout_prob)

    @property
    def condition_null(self) -> str:
        return self._config.condition_null

    @property
    def mask_value(self) -> float:
        return float(self._config.mask_value)

    @property
    def cfg_enabled(self) -> bool:
        """Whether a null (unconditional) condition was trained (``condition_dropout_prob > 0``) — the
        precondition for meaningful inference guidance. A guided solve with a field that never saw the
        null just re-derives the conditional velocity, so callers gate guidance on this."""
        return self.is_conditional and self.condition_dropout_prob > 0.0

    def null_condition_dict(self, condition_dict: MappedTensor) -> MappedTensor:
        """The ``mask_value`` null: every raw condition tensor filled with ``mask_value`` (kept in its
        original dtype so a categorical-index realm stays ``long``)."""
        mv = self.mask_value
        return {
            realm: torch.full_like(v, mv if v.is_floating_point() else int(mv))
            for realm, v in condition_dict.items()
        }

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
        self, encoded_condition: torch.Tensor | None, encoded_source: torch.Tensor | None
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

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        condition_dict: MappedTensor | None = None,
        source: torch.Tensor | None = None,
        condition_mask: MappedTensor | None = None,
        *,
        force_uncond: bool = False,
    ) -> torch.Tensor:
        # Inference/solver path: use the condition **mean** embedding (a stochastic encoder's inference
        # is its mean, i.e. encoder_noise = 0). Training reparameterizes upstream and calls
        # :meth:`velocity_from_embedding` directly.
        #
        # ``force_uncond`` selects the unconditional velocity v(x, t, null) for classifier-free guidance:
        # ``mask_value`` nulls the raw condition before encoding; ``zero_embedding`` zeros the pooled
        # embedding after. Both are no-ops for an unconditional field.
        if force_uncond and self.is_conditional and condition_dict is not None and self.condition_null == "mask_value":
            condition_dict = self.null_condition_dict(condition_dict)
        mean, _ = self.condition_stats(condition_dict, condition_mask=condition_mask)
        if force_uncond and mean is not None and self.condition_null == "zero_embedding":
            mean = torch.zeros_like(mean)
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

    def to_config(self) -> MLPVelocityConfig:
        # to_dict raises the "runtime-only custom combiner" / "custom pooling" errors; reuse that single
        # guard and rebuild a fresh, defensively-copied config from the portable dict.
        return MLPVelocityConfig.from_dict(self._config.to_dict())

    @classmethod
    def from_config(cls, config: MLPVelocityConfig) -> MLPVelocity:
        return config.build(VelocityFieldContext())

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
