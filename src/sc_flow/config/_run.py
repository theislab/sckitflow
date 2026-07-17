"""Structured run configuration for :class:`~sc_flow._model.SCFlow`.

``RunConfig`` is the single source of truth for constructing and training a
model. YAML, the fluent builder, and legacy keyword arguments all funnel into
it and through one validation path (see :meth:`SCFlow.from_config`).

The dataclasses are OmegaConf-friendly (primitives, ``Optional``, and
``dict[str, Any]`` blobs) so a run is fully serialisable. The one rule: the
``AnnData`` itself is never part of the config — it is passed at fit time.
Anything method-specific lives in the opaque ``MethodConfig.config`` blob,
validated by the method's own ``config_cls`` (see
:class:`~sc_flow.config._capabilities.MethodCapabilities`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["DataConfig", "MethodConfig", "TrainerConfig", "RunConfig"]


@dataclass
class DataConfig:
    """Data-setup settings (passed to ``SCFlow`` construction as the data context).

    Holds *settings only*; the ``AnnData`` is supplied at construction/fit time.
    """

    view_on_condition_space: bool = False
    condition_state_key: str | None = None
    #: Keyword arguments forwarded to ``DataManager(**datamanager)``. Callable /
    #: object-valued knobs (e.g. ``state_transform``) are given as
    #: ``{"kind": ..., ...}`` specs and resolved during ``from_config``.
    datamanager: dict[str, Any] = field(default_factory=dict)


@dataclass
class MethodConfig:
    """Which method to build and its opaque method-specific config."""

    method_id: str | None = None
    #: Fully-qualified ``"pkg.module:Class"`` path, as an alternative to ``method_id``.
    method_target: str | None = None
    #: Opaque to the framework; validated against the method's ``config_cls``.
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainerConfig:
    """Training-loop settings.

    Shared by the native loop (:meth:`SCFlow.train`) and the optional Lightning
    backend. ``framework`` selects which loop runs. The step/validation knobs are
    canonical and map onto Lightning: ``n_train_steps`` → ``max_steps`` and
    ``valid_freq`` → ``val_check_interval``; ``device`` → ``accelerator``/``devices``.
    The remaining fields are Lightning-facing (ignored by the native loop).
    """

    #: Which training loop to use: ``"native"`` (default) or ``"lightning"``.
    framework: str = "native"
    #: Single source of truth for the device; propagated into method construction
    #: and mapped to Lightning ``accelerator``/``devices``.
    device: str = "cpu"
    n_train_steps: int = 100_000
    train_batch_size: int = 128
    valid_freq: int = 1_000
    val_max_n_obs: int = 10_000
    train_sampler_kwargs: dict[str, Any] = field(default_factory=dict)
    val_sampler_kwargs: dict[str, Any] = field(default_factory=dict)
    #: Callback specs as ``{"kind"|"class_path": ..., ...}``, resolved during ``from_config``.
    callbacks: list[dict[str, Any]] = field(default_factory=list)
    #: Metric specs keyed by name, each ``{"kind"|"class_path": ..., ...}``.
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)

    # --- Lightning-facing (used when framework == "lightning") ---
    precision: str | None = None
    gradient_clip_val: float | None = None
    accumulate_grad_batches: int = 1
    log_every_n_steps: int | None = None
    #: Escape hatch: extra kwargs passed verbatim to ``lightning.Trainer``.
    trainer_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunConfig:
    """Complete specification of a training run."""

    data: DataConfig = field(default_factory=DataConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
    #: Keyword arguments for :class:`~sc_flow.methods._opt.OptimConfig`.
    optim: dict[str, Any] = field(default_factory=dict)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    seed: int | None = None
