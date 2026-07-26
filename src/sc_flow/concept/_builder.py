"""``FoundationModel`` — the contrastive-pretraining family builder (peer to ``FlowMatching``).

Same contract as :class:`sc_flow.FlowMatching`: from a ``recipe`` it builds the Lightning-native pieces —
``.module`` (a :class:`sc_flow.training.TrainingModule`), ``.datamodule``, ``.callbacks`` — and ``.save``\\s a
portable bundle. ``myapp.train`` (cf-train) drives it with a plain ``lightning.Trainer.fit`` exactly like the
flow path; the model family is chosen upstream by ``model.family``.

Structure is backbone / head / task so fine-tuning is a recipe change, not new plumbing:
``task="contrastive"`` pairs a :class:`ContrastiveHead` + the CLIP objective (pretrain); a future
``task="classify"`` would swap the head + objective over the *same* backbone (optionally
``pretrained=<dir>`` + ``freeze_backbone=True``).

recipe = {
  "data":      {"adata": <AnnData>, "species": "hsapiens"},   # in-memory source (+ optional "vocab_genes")
  "backbone":  {"dim_model": 512, "n_layers": 8, ...},        # gene_encoder config MINUS n_tokens (from data)
  "objective": {"logit_scale_init": 3.0, "max_logit_scale": 100.0},
  "sampler":   {"batch_size": 256, "max_tokens": 1024},
  "trainer":   {"lr": 1e-4},
  "task": "contrastive",           # pretrain (default)
  "pretrained": None,              # dir with a saved backbone to warm-start (fine-tune)
  "freeze_backbone": False,
  "seed": 0,
}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sc_flow.concept._data import FoundationDataModule
from sc_flow.concept._encoder import GeneEncoderConfig
from sc_flow.concept._head import ContrastiveHead, ContrastiveModel
from sc_flow.concept._objective import ContrastiveObjectiveConfig
from sc_flow.concept._vocab import GeneVocab
from sc_flow.training import TrainingModule

__all__ = ["FoundationModel"]

FORMAT_VERSION = 1


class FoundationModel:
    """Builder: recipe -> (LightningModule, LightningDataModule, callbacks). See the module docstring."""

    def __init__(self, recipe: dict[str, Any]) -> None:
        task = recipe.get("task", "contrastive")
        if task != "contrastive":
            raise NotImplementedError(
                f"task={task!r} not implemented yet; 'contrastive' is the pretrain task. Fine-tuning tasks "
                f"(classify/decode) are the documented extension: swap head + objective over the same backbone."
            )
        data = recipe["data"]
        adata = data["adata"]
        genes = data.get("vocab_genes") or list(adata.var_names)
        self._vocab = GeneVocab(genes, species=data.get("species", "unknown"))

        # Backbone (a Component) — sized by the vocab; the rest of its knobs come from the recipe.
        self._backbone_cfg = GeneEncoderConfig(n_tokens=self._vocab.n_tokens, **recipe.get("backbone", {}))
        backbone = self._backbone_cfg.build()
        if recipe.get("pretrained"):
            self._load_backbone(backbone, Path(recipe["pretrained"]))
        if recipe.get("freeze_backbone"):
            for p in backbone.parameters():
                p.requires_grad_(False)

        # Objective (a Component) + its head; compose into the trainable model.
        self._obj_cfg = ContrastiveObjectiveConfig(**recipe.get("objective", {}))
        objective = self._obj_cfg.build()
        head = ContrastiveHead(self._obj_cfg.logit_scale_init)
        self._model = ContrastiveModel(backbone, head)

        t, s = recipe.get("trainer", {}), recipe.get("sampler", {})
        self._module = TrainingModule(self._model, objective, lr=float(t.get("lr", 1e-4)))
        self._datamodule = FoundationDataModule(
            adata,
            self._vocab,
            batch_size=int(s.get("batch_size", 256)),
            max_tokens=int(s.get("max_tokens", 1024)),
            seed=int(recipe.get("seed", 0)),
            num_workers=int(s.get("num_workers", 0)),
        )
        self._callbacks: list = []  # kNN cell-type probe validation is the next addition

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
    def metrics_history(self) -> dict:
        return {}  # kNN cell-type probe validation (the next addition) will populate this

    @property
    def trainer_overrides(self) -> dict:
        return {}  # no bespoke eval cadence yet; the app's base Trainer is used as-is

    @property
    def vocab(self) -> GeneVocab:
        return self._vocab

    def _load_backbone(self, backbone, path: Path) -> None:
        from safetensors.torch import load_file

        weights = load_file(str(path / "model.safetensors"))
        # accept either a bare backbone state_dict or a full ContrastiveModel bundle
        bb = {k[len("backbone.") :]: v for k, v in weights.items() if k.startswith("backbone.")}
        backbone.load_state_dict(bb or weights, strict=False)

    def save(self, out: str | Path) -> None:
        """Persist the portable bundle: safetensors weights + a ``config.json`` of Component specs + vocab."""
        from safetensors.torch import save_file

        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        save_file(self._model.state_dict(), str(out / "model.safetensors"))
        (out / "config.json").write_text(
            json.dumps(
                {
                    "format": FORMAT_VERSION,
                    "family": "foundation",
                    "task": "contrastive",
                    "backbone": self._backbone_cfg.to_spec(),  # {type, version, config}
                    "objective": self._obj_cfg.to_spec(),
                    "species": self._vocab.species,
                    "vocab_genes": list(self._vocab.gene_ids),  # rebuild the exact token space on load
                },
                indent=2,
            )
        )
