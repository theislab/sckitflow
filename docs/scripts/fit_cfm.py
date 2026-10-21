"""Full training pipeline for a Conditional Flow Matching (CFM) model.

Wires together the components needed to train a Conditional Flow Matching
model on single-cell data with an MLP-parameterised velocity field. Designed
to be invoked via Hydra as the entry point of a training run:

    python -m <package>.fit_cfm [+config overrides]

(e.g. ``python -m sckitflow.training.fit_cfm`` — the ``-m`` flag takes a
module path, not a file; adjust to the module's actual location.)

Pipeline (in the order ``main`` executes them)
----------------------------------------------
1. Seed everything from ``config.run.method_seed`` (``lightning.seed_everything``).
2. Load AnnData from disk (``read_adatas``; paths in ``config.paths``).
3. Build a ``CombinationSplitter`` from ``config.splitter`` (control/train/val/test
   splits) and a ``FlowDataModule`` from ``config.datamanager`` + ``config.datamodule``.
4. Build the MLP velocity field (``MLPVelocityConfig`` from ``config.module``).
5. Build the base training method (``CFMTrainingConfig``), wrap it in a
   ``MatchedTrainingConfig`` that pairs source/target samples according to
   ``config.coupling``, and build the inference method (``ODEInferenceConfig``).
   Probability path and samplers arrive as portable ``*Config`` objects so
   ``Model.save_pretrained`` can serialize them; each method resolves them to a callable
   inside ``build``.
6. Build a ``torch.optim`` optimizer and validation metrics
   (``EnergyDistance``, ``MaximumMeanDiscrepancy``) from ``config.metrics``.
7. Build a ``TrainingPlan`` (the ``LightningModule`` wrapper) from
   ``config.training_plan``, wiring together the training/inference methods,
   metrics, optimizer, and transforms.
8. Configure ``WandbLogger`` + ``CSVLogger``, callbacks (``EarlyStopping``,
   ``GradientAccumulationScheduler``, ``LearningRateMonitor``, ``ModelCheckpoint``),
   and the ``lightning.pytorch.Trainer``.
9. ``trainer.fit(...)``.
10. Persist the model via ``Model.save_pretrained`` (writes ``config.json`` +
    ``module/model.safetensors``). Training checkpoints are written separately by
    the ``ModelCheckpoint`` callback during ``trainer.fit``. The fitted transform
    is persisted separately (``transform.json`` + ``transform_params.npz``) via
    ``save_transform``, since ``ModelConfig`` does not yet carry a ``transform``
    field. How the run went (optimizer, seeds, batch size) is this config's, not
    the model's.
11. Return the last validation ``edist`` from ``trainer.callback_metrics``
    (fallback: ``+inf``) for sweeps / shell exit codes. The key is
    ``f"{val_name}/edist"`` where ``val_name`` is the first entry of
    ``datamodule.val_names`` (default ``"val"`` → ``"val/edist"``).

Key concepts
------------
- **Conditional Flow Matching (CFM)**: learns a velocity field v(x, t | c) that
  transports a source distribution to a target distribution conditioned on c.
- **Probability path**: defines the interpolation between source and target
  samples. Supported: ``linear_dirac``, ``linear_gaussian``,
  ``schrodinger_bridge``, ``variance_preserving``. See
  ``sckitflow.core.probability_paths``.
- **Velocity field**: the network that predicts the time derivative along the
  path. Here an MLP with separate state/time/condition encoders.
- **Coupling**: at training time, pairs source samples with target samples
  before regression. Selected by ``config.coupling.method``: ``independent``,
  ``exact``, ``sinkhorn``, ``partial``, ``unbalanced``,
  ``gromov-wasserstein``, ``fused``. See ``sckitflow.core.coupling``.
- **Sampler**: time and noise draws during training / inference.
  ``time_sampler`` and ``noise_sampler`` in the config are ``SamplerConfig``
  objects (portable — the run's spec serializes them). See
  ``sckitflow.core._samplers``.
- **Transform**: optional pre-processing of the ``sample_rep`` (or ``.X`` when
  ``sample_rep is None``), fitted on the training split only. Configured via
  ``config.transform``. Persisted to ``run_dir`` alongside the run.
- **Inference**: integrates the learned ODE from t=0 to t=1 with ``torchdiffeq``,
  configured via ``ODEInferenceConfig``.

Configuration
-------------
The full YAML is expected at ``<module_dir>/../config/fit_cfm.yaml`` (Hydra).
See that file for every key, its meaning, and its default. The coupling
selection and its hyperparameters live under ``config.coupling``.
``hydra.job.chdir: false`` is recommended so relative paths
(``save.run_dir``, ``callbacks.model_checkpoint.dirpath``, logger
``save_dir``) resolve from the launch directory rather than Hydra's
``outputs/<date>/<time>/`` tree.

Outputs
-------
- Checkpoints: ``config.callbacks.model_checkpoint.dirpath`` (written by the
  ``ModelCheckpoint`` callback during training; best-``val_edist`` weights
  plus ``last.ckpt``).
- Loggers: ``config.loggers.{wandb,csv}.save_dir``.
- Saved model: ``config.save.run_dir`` (via ``Model.save_pretrained``; contains
  ``config.json`` — the portable ``ModelConfig`` — and ``module/model.safetensors``).
  Reload it with ``sckitflow.Model.from_pretrained(run_dir, adata)`` for
  **inference**.
- Fitted transform: ``config.save.run_dir`` (via ``save_transform``; contains
  ``transform.json`` — the portable ``TransformConfig`` spec — and
  ``transform_params.npz`` — the fitted numpy arrays). Loadable via
  ``sckitflow.preprocessing.transforms.load_transform`` for applying the same
  transform in a different process.
- Return value: last ``<val_name>/edist`` from
  ``trainer.callback_metrics`` (``+inf`` if validation never ran).
"""

from typing import Any

import anndata as ad
import hydra
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    EarlyStopping,
    GradientAccumulationScheduler,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger, CSVLogger
import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.sparse import issparse

import torch

from sckitflow import (
    Model,
    ModelConfig,
    get_logger,
    IsotropicGaussianNoiseSamplerConfig,
    UniformTimeSamplerConfig,
)
from sckitflow.core.coupling import (
    IndependentCouplingConfig,
    ExactCouplingConfig,
    SinkhornCouplingConfig,
    PartialCouplingConfig,
    UnbalancedCouplingConfig,
    GromovWassersteinCouplingConfig,
    FusedGromovWassersteinCouplingConfig,
)
from sckitflow.core.methods import CFMTrainingConfig, MatchedTrainingConfig, ODEInferenceConfig
from sckitflow.core.metrics import EnergyDistance, MaximumMeanDiscrepancy
from sckitflow.core.nn import MLPVelocityConfig
from sckitflow.core.probability_paths import (
    LinearDiracProbabilityPathConfig,
    LinearGaussianProbabilityPathConfig,
    SchrodingerBridgeProbabilityPathConfig,
    VariancePreservingDiracProbabilityPathConfig,
)
from sckitflow.data import CombinationSplitterConfig, DataManagerConfig, FlowDataModule
from sckitflow.preprocessing.transforms import (
    IdentityTransformConfig,
    PCATransformConfig,
    WhiteningTransformConfig,
    ZScoreTransformConfig,
    save_transform,
)
from sckitflow.trainer import TrainingPlan


logger = get_logger(__name__)

EDIST_KEY = "edist"
MMD_KEY = "mmd"
ORIGINAL_REP_KEY = "original_rep"


def resolve_omegaconf_to_dictionary(conf_dict: dict[str, Any] | None | DictConfig) -> dict[str, Any]:
    """Resolves the omegaconf ``DictConfig`` to a plain dictionary."""
    out_dict = {}
    if conf_dict is not None:
        out_dict = conf_dict
        if isinstance(out_dict, dict):
            out_dict = OmegaConf.create(out_dict)
        out_dict = OmegaConf.to_container(out_dict, resolve=True)
    return out_dict


@hydra.main(config_path="../configs", config_name="fit_cfm", version_base=None)
def main(config: DictConfig) -> float:
    """Main entrypoint for CFM training."""
    # Seed everything with lightning
    # This is the master seeding, and it will be also saved to the run
    # for further reproducibility.
    logger.info(f"Setting the global run seed to {config.run.method_seed}")
    seed_everything(seed=config.run.method_seed, workers=True)

    # Load data from disk using the utility function.
    # Please override this function to allow custom operations on the
    # data after loading; at the moment, only vanilla loading
    # with ``anndata`` is performed, and no other transformation is
    # applied to the data.
    logger.info(f"Reading target adata from {config.paths.target_adata}.")
    adata = ad.read_h5ad(config.paths.target_adata)
    logger.info(f"Target adata read: {adata}")

    # Prepare splitter for training.
    # The splitter defines control, train, validation and test splits
    # and writes a column in the input ``ad.AnnData``. We should always
    # fit any data transformation only on the training cases.
    logger.info(f"Initializing splitter with seed {config.run.splitter_seed}.")
    # If the input already carries the split column -- for example because it
    # was produced by ``prepare_panel_adata.py``, which writes an 80/20
    # ``sklearn`` ``train_test_split`` into ``obs["split"]`` -- reuse it and
    # skip the splitter entirely. ``Splitter.split`` refuses to overwrite an
    # existing split column (a split is a hold-out decision), so redrawing
    # here would either fail or silently change which observations are held
    # out. Drop the column to force the configured splitter to run.
    split_key = config.splitter.split_key
    if split_key in adata.obs.columns:
        label_counts = adata.obs[split_key].value_counts().to_dict()
        logger.info(
            f"adata.obs already has a {split_key!r} column; reusing the existing split "
            f"and skipping the splitter. Drop `adata.obs[{split_key!r}]` to re-split. "
            f"Existing label counts: {label_counts}"
        )
    else:
        logger.info(f"Initializing splitter with seed {config.run.splitter_seed}.")
        splitter_rng = np.random.default_rng(config.run.splitter_seed)
        splitter_cfg = CombinationSplitterConfig(
            split_key=config.splitter.split_key,
            group_keys=config.splitter.group_keys,
            always_train_keys=config.splitter.always_train_keys,
            control_key=config.splitter.control_key,
            control_value=config.splitter.control_value,
            test_fraction=config.splitter.test_fraction,
            train_label=config.splitter.train_label,
            test_label=config.splitter.test_label,
            control_label=config.splitter.control_label,
        )
        splitter = splitter_cfg.build(rng=splitter_rng)

        # Split the data upfront. This is required for optional transformations
        # on the data -- their parameters should be inferred exclusively from
        # the training data, therefore we need to explicitly split the input
        # data to avoid information leakage.
        adata = splitter.split(adata, copy=False)

    # Subsetting the data to train only, so that we can then infer the
    # parameters for the required transformations without leaking information
    # from the validation and test sets.
    train_mask = adata.obs[config.splitter.split_key] == config.splitter.train_label
    training_adata = adata[train_mask]

    # Applying transformations to the data. First, we need to infer the
    # parameters from the training data, then we have to transform all of the
    # data using such parameters.
    # TODO: Consider how to make the transformations persistent.
    if config.transform.name == "identity":
        transform_cfg = IdentityTransformConfig()
    elif config.transform.name == "pca":
        transform_cfg = PCATransformConfig(n_components=config.transform.n_components)
    elif config.transform.name == "zscore":
        transform_cfg = ZScoreTransformConfig(epsilon=config.transform.epsilon)
    elif config.transform.name == "whitening":
        transform_cfg = WhiteningTransformConfig(
            method=config.transform.method,
            epsilon=config.transform.epsilon,
        )
    else:
        raise ValueError(f"Invalid transform {config.transform.name}")
    transform = transform_cfg.build()

    # Fitting the transformation parameters on the training split only.
    transform = transform.fit_adata(
        training_adata,
        obsm=config.datamanager.sample_rep,
    )

    # Now we need to store the original representation in a new key, so that
    # we can retrieve it back later after transformation.
    if config.datamanager.sample_rep is not None:
        if config.datamanager.sample_rep not in adata.obsm_keys():
            raise ValueError(
                f"Representation {config.datamanager.sample_rep!r} not found in "
                f"adata.obsm; available keys: {list(adata.obsm_keys())}."
            )
        X_original_rep = adata.obsm[config.datamanager.sample_rep]
    else:
        X_original_rep = adata.X

    # ``obsm`` requires dense arrays, and every transform expects a 2-D
    # ``np.ndarray``. Densify once here so both the persistence call and the
    # transform application see a consistent representation.
    if issparse(X_original_rep):
        X_original_rep = X_original_rep.toarray()
    else:
        X_original_rep = np.asarray(X_original_rep)
    adata.obsm[ORIGINAL_REP_KEY] = X_original_rep

    # We then apply the transformation and store the resulting transformed
    # values back where they came from, so that they can still be read from
    # the rest of the script.
    X_transformed = transform.transform(X_original_rep)
    if config.datamanager.sample_rep is not None:
        adata.obsm[config.datamanager.sample_rep] = X_transformed
    else:
        adata.X = X_transformed

    # Prepare data manager.
    # The ``DataManager`` is never materialized directly here: it lives
    # in the ``datamodule``, of which it is an attribute.
    logger.info("Initializing datamanager.")
    dm_cfg = DataManagerConfig(
        sample_rep=config.datamanager.sample_rep,
        categorical_conditions=config.datamanager.categorical_conditions,
        categorical_reps=config.datamanager.categorical_reps,
        continuous_conditions=config.datamanager.continuous_conditions,
        control_values_dict=config.datamanager.control_values_dict,
        matched_pairs=config.datamanager.matched_pairs,
        split_by=config.datamanager.split_by,
        condition_state_key=config.datamanager.condition_state_key,
        target_categorical_covs_dict=config.datamanager.target_categorical_covs_dict,
        target_continuous_covs=config.datamanager.target_continuous_covs,
        groups=config.datamanager.groups,
        groups_reps=config.datamanager.groups_reps,
        groups_encoding=config.datamanager.groups_encoding,
        rep_lin=config.datamanager.rep_lin,
        source_rep_quad=config.datamanager.source_rep_quad,
        target_rep_quad=config.datamanager.target_rep_quad,
        use_linear=config.datamanager.use_linear,
    )

    # Initialize datamodule.
    # The loaders' sampling schedule is seeded from ``config.run.loader_seed`` for reproducibility.
    logger.info(f"Initializing datamodule with seed {config.run.loader_seed}")
    datamodule_rng = np.random.default_rng(config.run.loader_seed)
    datamodule = FlowDataModule.from_adata(
        adata,
        dm_cfg.build(),
        train_split=config.datamodule.train_split,
        n_train_steps=config.datamodule.n_train_steps,
        batch_size=config.datamodule.batch_size,
        dtype=getattr(torch, config.datamodule.dtype),
        loader_kwargs={
            **resolve_omegaconf_to_dictionary(config.datamodule.loader_kwargs),
            "seed": int(datamodule_rng.integers(2**63)),
        },
    )

    # Initialize neural network.
    # The dimensionalities are automatically inferred from the
    # data_dims object, linked to the datamodule.
    logger.info("Initializing neural velocity field.")
    module_cfg = MLPVelocityConfig(
        encode_state=config.module.encode_state,
        encode_time=config.module.encode_time,
        time_features_id=config.module.time_features_id,
        num_time_features=config.module.num_time_features,
        max_period=config.module.max_period,
        state_encoder_output_dim=config.module.state_encoder_output_dim,
        time_encoder_output_dim=config.module.time_encoder_output_dim,
        state_encoder_mlp_kwargs=resolve_omegaconf_to_dictionary(config.module.state_encoder_mlp_kwargs),
        time_encoder_mlp_kwargs=resolve_omegaconf_to_dictionary(config.module.time_encoder_mlp_kwargs),
        vf_decoder_mlp_kwargs=resolve_omegaconf_to_dictionary(config.module.vf_decoder_mlp_kwargs),
        conditioning_id=config.module.conditioning_id,
        conditioning_kwargs=resolve_omegaconf_to_dictionary(config.module.conditioning_kwargs),
        condition_encoder_input_layers=config.module.condition_encoder_input_layers,
        condition_encoder_output_dim=config.module.condition_encoder_output_dim,
        condition_encoder_pooling_mode=config.module.condition_encoder_pooling_mode,
        condition_encoder_pooling_kwargs=resolve_omegaconf_to_dictionary(
            config.module.condition_encoder_pooling_kwargs
        ),
        condition_encoder_pooling_proj_dim=config.module.condition_encoder_pooling_proj_dim,
        condition_encoder_pooling_proj_bias=config.module.condition_encoder_pooling_proj_bias,
        condition_encoder_output_layers_kwargs=resolve_omegaconf_to_dictionary(
            config.module.condition_encoder_output_layers_kwargs
        ),
        source_encoder_mlp_kwargs=None
        if config.module.source_encoder_mlp_kwargs is None
        else resolve_omegaconf_to_dictionary(config.module.source_encoder_mlp_kwargs),
        source_encoder_output_dim=config.module.source_encoder_output_dim,
    )
    module = module_cfg.build(datamodule.data_dims)

    # Initialize probability path
    logger.info("Initializing probability path.")
    if config.probability_path.name == "linear_dirac":
        probability_path_cfg = LinearDiracProbabilityPathConfig(sigma=config.probability_path.sigma)
    elif config.probability_path.name == "linear_gaussian":
        probability_path_cfg = LinearGaussianProbabilityPathConfig(sigma=config.probability_path.sigma)
    elif config.probability_path.name == "schrodinger_bridge":
        probability_path_cfg = SchrodingerBridgeProbabilityPathConfig(
            sigma=config.probability_path.sigma, eps=config.probability_path.eps
        )
    elif config.probability_path.name == "variance_preserving":
        probability_path_cfg = VariancePreservingDiracProbabilityPathConfig(sigma=config.probability_path.sigma)
    else:
        raise ValueError(f"Invalid probability path name {config.probability_path.name}.")

    # Initialize time sampler
    logger.info("Initializing time sampler.")
    if config.time_sampler.name == "uniform":
        time_sampler_cfg = UniformTimeSamplerConfig()
    else:
        raise ValueError(f"Invalid time sampler name {config.time_sampler.name}.")

    # Initialize noise sampler
    logger.info("Initializing noise sampler.")
    if config.noise_sampler.name == "isotropic_gaussian":
        noise_sampler_cfg = IsotropicGaussianNoiseSamplerConfig()
    else:
        raise ValueError(f"Invalid noise sampler name {config.noise_sampler.name}.")

    # Initialize training module.
    logger.info("Initializing base training method.")
    base_training_cfg = CFMTrainingConfig(
        probability_path=probability_path_cfg,
        time_sampler=time_sampler_cfg,
        noise_sampler=noise_sampler_cfg,
        generate_from_noise=config.training.generate_from_noise,
    )

    # Initialize coupling configuration and matched training configurations.
    logger.info("Initializing coupling module.")
    if config.coupling.method == "independent":
        coupling_cfg = IndependentCouplingConfig()
    elif config.coupling.method == "exact":
        coupling_cfg = ExactCouplingConfig(
            scale_cost=config.coupling.scale_cost,
            num_iter_max=config.coupling.num_iter_max,
        )
    elif config.coupling.method == "sinkhorn":
        coupling_cfg = SinkhornCouplingConfig(
            scale_cost=config.coupling.scale_cost,
            num_iter_max=config.coupling.num_iter_max,
            reg=config.coupling.reg,
            stop_thr=config.coupling.stop_thr,
        )
    elif config.coupling.method == "partial":
        coupling_cfg = PartialCouplingConfig(
            scale_cost=config.coupling.scale_cost,
            num_iter_max=config.coupling.num_iter_max,
            reg=config.coupling.reg,
            stop_thr=config.coupling.stop_thr,
            m=config.coupling.m,
        )
    elif config.coupling.method == "unbalanced":
        coupling_cfg = UnbalancedCouplingConfig(
            scale_cost=config.coupling.scale_cost,
            num_iter_max=config.coupling.num_iter_max,
            reg=config.coupling.reg,
            stop_thr=config.coupling.stop_thr,
            reg_m=config.coupling.reg_m,
        )
    elif config.coupling.method == "gromov-wasserstein":
        coupling_cfg = GromovWassersteinCouplingConfig(
            scale_cost=config.coupling.scale_cost,
            epsilon=config.coupling.epsilon,
            max_iter=config.coupling.max_iter,
            tol=config.coupling.tol,
        )
    elif config.coupling.method == "fused":
        coupling_cfg = FusedGromovWassersteinCouplingConfig(
            scale_cost=config.coupling.scale_cost,
            epsilon=config.coupling.epsilon,
            max_iter=config.coupling.max_iter,
            tol=config.coupling.tol,
            alpha=config.coupling.alpha,
        )
    else:
        raise ValueError(f"Invalid coupling method {config.coupling.method}")

    # Initialize wrapped method for matched training.
    logger.info("Initializing matched wrapper.")
    training_cfg = MatchedTrainingConfig(
        method=base_training_cfg,
        coupling=coupling_cfg,
    )

    logger.info("Building training method.")
    training_method = training_cfg.build(
        module,
        dims=datamodule.data_dims,
    )

    # Initialize inference module.
    logger.info("Initializing inference method.")
    inference_cfg = ODEInferenceConfig(
        probability_path=probability_path_cfg,
        time_sampler=time_sampler_cfg,
        noise_sampler=noise_sampler_cfg,
        generate_from_noise=config.inference.generate_from_noise,
        solver_kwargs=resolve_omegaconf_to_dictionary(config.inference.solver_kwargs),
        return_trajectory=config.inference.return_trajectory,
        n_steps=config.inference.n_steps,
        n_samples=config.inference.n_samples,
    )
    logger.info("Building inference method.")
    inference_method = inference_cfg.build(module, dims=datamodule.data_dims)

    # Prepare optimizer and scheduler.
    logger.info("Initializing optimizer.")
    optimizer = getattr(torch.optim, config.optimizer.name)(
        module.parameters(), **resolve_omegaconf_to_dictionary(config.optimizer.kwargs)
    )
    lr_scheduler = None

    # Initialize metrics used to evaluate model's predictions during validation.
    logger.info("Initializing metrics.")
    edist = EnergyDistance(dist_sync_on_step=config.metrics.edist.dist_sync_on_step)
    mmd = MaximumMeanDiscrepancy(
        gammas=config.metrics.mmd.gammas,
        dist_sync_on_step=config.metrics.mmd.dist_sync_on_step,
    )
    metrics = {
        EDIST_KEY: edist,
        MMD_KEY: mmd,
    }

    # Initialize prediction transformations.
    # TODO: wire these to ``config.transforms.{pred,target}`` when that section
    # exists; leaving them ``None`` runs the plan without post-hoc transforms.
    pred_transform = None  # TODO

    # Initialize target transformations.
    target_transform = None  # TODO

    # Initialize training plan.
    logger.info("Initializing training plan.")
    plan = TrainingPlan(
        training_method,
        optimizer,
        lr_scheduler=lr_scheduler,
        lr_scheduler_interval=config.training_plan.lr_scheduler_interval,
        inference_method=inference_method,
        metrics=metrics,
        val_names=datamodule.val_names,
        pred_transform=pred_transform,
        target_transform=target_transform,
        seed=config.run.method_seed,
    )

    # Prepare wandb and csv loggers.
    logger.info("Initializing wandb logger.")
    wandb_logger = WandbLogger(
        name=config.loggers.wandb.name,
        save_dir=config.loggers.wandb.save_dir,
        offline=config.loggers.wandb.offline,
        id=config.loggers.wandb.id,
        anonymous=config.loggers.wandb.anonymous,
        version=config.loggers.wandb.version,
        project=config.loggers.wandb.project,
        log_model=config.loggers.wandb.log_model,
        experiment=config.loggers.wandb.experiment,
        prefix=config.loggers.wandb.prefix,
        **resolve_omegaconf_to_dictionary(config.loggers.wandb.init_kwargs),
    )
    logger.info("Initializing csv logger.")
    csv_logger = CSVLogger(
        config.loggers.csv.save_dir,
        name=config.loggers.csv.name,
        version=config.loggers.csv.version,
        prefix=config.loggers.csv.prefix,
        flush_logs_every_n_steps=config.loggers.csv.flush_logs_every_n_steps,
    )
    training_loggers = [wandb_logger, csv_logger]

    # Prepare callbacks.
    logger.info("Initializing early stopping callback.")
    early_stopping = EarlyStopping(
        monitor=config.callbacks.early_stopping.monitor,
        min_delta=config.callbacks.early_stopping.min_delta,
        patience=config.callbacks.early_stopping.patience,
        verbose=config.callbacks.early_stopping.verbose,
        mode=config.callbacks.early_stopping.mode,
        strict=config.callbacks.early_stopping.strict,
        check_finite=config.callbacks.early_stopping.check_finite,
        stopping_threshold=config.callbacks.early_stopping.stopping_threshold,
        divergence_threshold=config.callbacks.early_stopping.divergence_threshold,
        check_on_train_epoch_end=config.callbacks.early_stopping.check_on_train_epoch_end,
    )
    logger.info("Initializing gradient accumulation callback.")
    grad_accumulation = GradientAccumulationScheduler(scheduling=config.callbacks.grad_accumulation.scheduling)
    logger.info("Initializing learning rate monitor callback.")
    lr_monitor = LearningRateMonitor(
        logging_interval=config.callbacks.lr_monitor.logging_interval,
        log_momentum=config.callbacks.lr_monitor.log_momentum,
    )
    logger.info("Initializing model checkpoint callback.")
    model_checkpoint = ModelCheckpoint(
        dirpath=config.callbacks.model_checkpoint.dirpath,
        filename=config.callbacks.model_checkpoint.filename,
        monitor=config.callbacks.model_checkpoint.monitor,
        verbose=config.callbacks.model_checkpoint.verbose,
        save_last=config.callbacks.model_checkpoint.save_last,
        save_top_k=config.callbacks.model_checkpoint.save_top_k,
        save_weights_only=config.callbacks.model_checkpoint.save_weights_only,
        mode=config.callbacks.model_checkpoint.mode,
        auto_insert_metric_name=config.callbacks.model_checkpoint.auto_insert_metric_name,
        every_n_train_steps=config.callbacks.model_checkpoint.every_n_train_steps,
        train_time_interval=config.callbacks.model_checkpoint.train_time_interval,
        every_n_epochs=config.callbacks.model_checkpoint.every_n_epochs,
        save_on_train_epoch_end=config.callbacks.model_checkpoint.save_on_train_epoch_end,
    )
    callbacks = [early_stopping, grad_accumulation, lr_monitor, model_checkpoint]

    # Configuring the training plugins.
    plugins = None  # TODO

    # Initialize ``lightning.pytorch.Trainer`` object.
    logger.info("Initializing trainer.")
    trainer = Trainer(
        accelerator=config.trainer.accelerator,
        strategy=config.trainer.strategy,
        devices=config.trainer.devices,
        num_nodes=config.trainer.num_nodes,
        precision=config.trainer.precision,
        logger=training_loggers,
        callbacks=callbacks,
        fast_dev_run=config.trainer.fast_dev_run,
        max_epochs=config.trainer.max_epochs,
        min_epochs=config.trainer.min_epochs,
        max_steps=config.trainer.max_steps,
        min_steps=config.trainer.min_steps,
        max_time=config.trainer.max_time,
        limit_train_batches=config.trainer.limit_train_batches,
        limit_val_batches=config.trainer.limit_val_batches,
        limit_test_batches=config.trainer.limit_test_batches,
        limit_predict_batches=config.trainer.limit_predict_batches,
        overfit_batches=config.trainer.overfit_batches,
        val_check_interval=config.trainer.val_check_interval,
        check_val_every_n_epoch=config.trainer.check_val_every_n_epoch,
        num_sanity_val_steps=config.trainer.num_sanity_val_steps,
        log_every_n_steps=config.trainer.log_every_n_steps,
        enable_checkpointing=config.trainer.enable_checkpointing,
        enable_progress_bar=config.trainer.enable_progress_bar,
        enable_model_summary=config.trainer.enable_model_summary,
        gradient_clip_val=config.trainer.gradient_clip_val,
        gradient_clip_algorithm=config.trainer.gradient_clip_algorithm,
        deterministic=config.trainer.deterministic,
        benchmark=config.trainer.benchmark,
        inference_mode=config.trainer.inference_mode,
        use_distributed_sampler=config.trainer.use_distributed_sampler,
        profiler=config.trainer.profiler,  # TODO: this can be an object, resolve it somehow.
        detect_anomaly=config.trainer.detect_anomaly,
        barebones=config.trainer.barebones,
        plugins=plugins,
        sync_batchnorm=config.trainer.sync_batchnorm,
        reload_dataloaders_every_n_epochs=config.trainer.reload_dataloaders_every_n_epochs,
        default_root_dir=config.trainer.default_root_dir,
        enable_autolog_hparams=config.trainer.enable_autolog_hparams,
        model_registry=config.trainer.model_registry,
    )

    logger.info("Fitting the model.")
    trainer.fit(plan, datamodule=datamodule, ckpt_path=config.trainer.ckpt_path)

    # Save the trained model: its spec and the module's weights.
    model_cfg = ModelConfig(data=dm_cfg, module=module_cfg, training=training_cfg, inference=inference_cfg)
    model = Model(model_cfg, datamodule.dm, datamodule.data_dims, module, training_method, inference_method)
    logger.info(f"Saving model {config.save.run_dir}.")
    model.save_pretrained(config.save.run_dir, overwrite=config.run.allow_overwrite)

    # Persist the fitted transform alongside the model. Kept separate from
    # ``save_pretrained`` because ``ModelConfig`` does not yet carry a ``transform``
    # field — see ``save_transform``'s docstring in
    # ``sckitflow.preprocessing.transforms._persistence``.
    logger.info(f"Saving fitted transform to {config.save.run_dir}.")
    save_transform(config.save.run_dir, transform_cfg, transform)

    # Capture the edist from the latest validation step.
    val_edist_key = f"{datamodule.val_names[0]}/{EDIST_KEY}"  # e.g. "val/edist"
    last_val_edist = trainer.callback_metrics.get(val_edist_key, float("inf"))
    returned_metric = float(last_val_edist)
    logger.info(f"Finishing the run with edist of {returned_metric:.4f}")
    return returned_metric


if __name__ == "__main__":
    main()
