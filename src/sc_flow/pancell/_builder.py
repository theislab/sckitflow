"""``PanCellFlow`` — the pan-cell flow *composition* (a ``recipe -> builder``; **not** a registered family).

``PanCellFlow`` composes a **foundation** ``sc_flow.gene_encoder`` (the state-encoder slot — optionally
``pretrained`` from a saved foundation bundle and ``freeze``d) with a rectified-flow velocity + FM objective:
one encoder gives flow a shared latent across gene panels. It composes the two paradigms rather than being a
third one, so it is **not** a ``scfit.families`` entry point — construct it directly (``PanCellFlow(recipe)``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scfoundations._encoder import GeneEncoderConfig
from scfoundations._vocab import GeneVocab
from sc_flow.pancell._data import PanCellDataModule
from sc_flow.pancell._model import PanCellFlowModel, VelocityMLPConfig
from sc_flow.pancell._objective import LinearFMObjectiveConfig
from sc_flow.training import TrainingModule

__all__ = ["PanCellFlow"]


class PanCellFlow:
    """Builder: recipe -> (.module, .datamodule, .callbacks, .save). See the module docstring."""

    def __init__(self, recipe: dict[str, Any]) -> None:
        d = recipe["data"]
        adata = d["adata"]
        control_key = d.get("control_key", "is_control")
        self._vocab = GeneVocab(d.get("vocab_genes") or list(adata.var_names))

        enc = dict(recipe.get("state_encoder", {}))
        pretrained, freeze = enc.pop("pretrained", None), enc.pop("freeze", False)
        self._encoder_cfg = GeneEncoderConfig(n_tokens=self._vocab.n_tokens, **enc)
        encoder = self._encoder_cfg.build()
        if pretrained:
            self._load_encoder(encoder, Path(pretrained))
        if freeze:
            for p in encoder.parameters():
                p.requires_grad_(False)

        self._velocity_cfg = VelocityMLPConfig(dim=encoder.dim_model, **recipe.get("velocity", {}))
        self._obj_cfg = LinearFMObjectiveConfig(**recipe.get("objective", {}))
        model = PanCellFlowModel(encoder, self._velocity_cfg.build())
        self._model = model

        s, t = recipe.get("sampler", {}), recipe.get("trainer", {})
        self._module = TrainingModule(model, self._obj_cfg.build(), lr=float(t.get("lr", 1e-3)))
        is_control = adata.obs[control_key].to_numpy().astype(bool)
        self._datamodule = PanCellDataModule(
            adata[is_control], adata[~is_control], self._vocab,
            batch_size=int(s.get("batch_size", 128)), max_tokens=int(s.get("max_tokens", 256)),
            steps_per_epoch=int(s.get("steps_per_epoch", 2000)), seed=int(recipe.get("seed", 0)),
        )
        self._callbacks: list = []

    @property
    def module(self):
        return self._module

    @property
    def datamodule(self):
        return self._datamodule

    @property
    def callbacks(self) -> list:
        return list(self._callbacks)

    @property
    def model(self):
        return self._model

    def _load_encoder(self, encoder, path: Path) -> None:
        from safetensors.torch import load_file

        w = load_file(str(path / "model.safetensors"))
        bb = {k[len("backbone.") :]: v for k, v in w.items() if k.startswith("backbone.")}
        encoder.load_state_dict(bb or w, strict=False)

    def save(self, out: str | Path) -> None:
        from safetensors.torch import save_file

        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        save_file(self._model.state_dict(), str(out / "model.safetensors"))
        (out / "config.json").write_text(json.dumps({
            "format": 1, "family": "pancell",
            "state_encoder": self._encoder_cfg.to_spec(),
            "velocity": self._velocity_cfg.to_spec(),
            "objective": self._obj_cfg.to_spec(),  # nests the probability_path spec
            "vocab_genes": list(self._vocab.gene_ids),
        }, indent=2))
