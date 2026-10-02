"""Turning a run of predictions back into an ``AnnData``.

Free functions, not methods: reassembly is a pure transform of
``(predictions, dimensionalities)`` and holds no state of its own.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import fields
from typing import Any

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from tqdm import tqdm

from sckitflow._random import generators
from sckitflow.core._types import PredictionData, StepData, concatenate_predictions
from sckitflow.core.methods._base import SupportsInference
from sckitflow.data._datamodule import FlowDataModule
from sckitflow.data._dims import DataDimensions

__all__ = ["prediction_record", "predictions_to_adata", "predict_adata"]


def _predict_empty(data_dims: DataDimensions, return_raw: bool) -> AnnData | tuple[AnnData, None]:
    """Returns empty anndata for prediction."""
    empty_adata = AnnData(
        X=np.empty((0, len(data_dims.feature_names))),
        var=pd.DataFrame(index=data_dims.feature_names),
    )
    return empty_adata if not return_raw else (empty_adata, None)


def _pred_obs_from_leaf(group_cols: tuple[str, ...], leaf: tuple, pred_obj: PredictionData) -> pd.DataFrame:
    """Rebuild a group's obs rows from its ``leaf`` (the ``group_by`` value tuple), one per predicted observation."""
    n_pred_obs = pred_obj.X.shape[0] if getattr(pred_obj, "X", None) is not None else 1
    return pd.DataFrame({col: np.repeat(val, n_pred_obs) for col, val in zip(group_cols, leaf, strict=True)})


def _get_pred_traj(pred_obj: PredictionData) -> np.ndarray | None:
    if pred_obj.traj is None:
        return None

    n_obs = pred_obj.X.shape[0]
    traj_np = _to_numpy(pred_obj.traj)

    if traj_np.ndim == 2 and traj_np.shape[0] == n_obs:
        return traj_np
    elif traj_np.ndim == 3 and traj_np.shape[1] == n_obs:
        return np.transpose(traj_np, (1, 0, 2))
    elif traj_np.ndim == 4 and traj_np.shape[2] == n_obs:
        return np.transpose(traj_np, (2, 0, 1, 3))
    else:
        raise ValueError(
            "Trajectory array has incompatible shape for AnnData.obsm: "
            f"got {traj_np.shape}, expected first dimension to equal "
            f"n_obs ({n_obs}) or, for 3D trajectories, second "
            "dimension to equal n_obs so it can be transposed from "
            "(n_time_steps, n_obs, n_features) to "
            "(n_obs, n_time_steps, n_features)."
        )


def _get_pred_raw_samples(pred_obj: PredictionData) -> np.ndarray | None:
    raw_samples = getattr(pred_obj, "raw_samples", None)
    if raw_samples is None:
        return None

    X = getattr(pred_obj, "X", None)
    if X is None:
        raise ValueError("Prediction object should have the .X attribute.")
    n_obs = X.shape[0]

    samples_np = _to_numpy(raw_samples)
    if samples_np.ndim == 2 and samples_np.shape[0] == n_obs:
        return samples_np
    elif samples_np.ndim == 3 and samples_np.shape[1] == n_obs:
        return np.transpose(samples_np, (1, 0, 2))
    else:
        raise ValueError(
            "Samples array has incompatible shape for AnnData.obsm: "
            f"got {samples_np.shape}, expected data of shape "
            f"(n_obs, n_features) or (n_samples, n_obs, n_features)"
        )


def _get_pred_obsm_dict(
    step_data: StepData, pred_obj: PredictionData, cont_keys: tuple[str, ...]
) -> dict[str, np.ndarray]:
    obsm_dict: dict[str, np.ndarray] = {}
    traj = _get_pred_traj(pred_obj)
    if traj is not None:
        obsm_dict["trajectory"] = traj
    raw_samples = _get_pred_raw_samples(pred_obj)
    if raw_samples is not None:
        obsm_dict["raw_samples"] = raw_samples

    condition = step_data["target_condition_data"] or {}
    response = step_data["target_response_data"] or {}
    for key in cont_keys:
        if key in condition:
            obsm_dict[key] = _to_numpy(condition[key])
        elif key in response:
            obsm_dict[key] = _to_numpy(response[key])
    return obsm_dict


def _aggregate_nodes_pred(
    data_dims: DataDimensions,
    all_preds: list[PredictionData],
    all_obs: list[pd.DataFrame],
    all_obsm: dict[str, list[np.ndarray]],
    return_raw: bool = False,
) -> AnnData | tuple[AnnData, PredictionData]:
    merged_pred = concatenate_predictions(all_preds)

    X_np = _to_numpy(merged_pred.X)

    obs_final = pd.concat(all_obs, axis=0, ignore_index=True)
    obs_final.index = obs_final.index.astype(str)

    obsm_final = {k: np.concatenate(v, axis=0) for k, v in all_obsm.items()}

    pred_adata = AnnData(X=X_np, obs=obs_final, var=pd.DataFrame(index=data_dims.feature_names), obsm=obsm_final)

    if return_raw:
        return pred_adata, merged_pred

    return pred_adata


def _to_numpy(tensor: Any) -> np.ndarray:
    """Convert a torch tensor (or array-like) to a numpy array."""
    if tensor is None:
        return None

    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.array(tensor)


def prediction_record(
    loader: Any,
    step_data: StepData,
    leaf: tuple,
    preds: PredictionData,
) -> dict[str, Any]:
    """Shapes one group's prediction into what :func:`predictions_to_adata` consumes.

    ``PredictionData`` is a frozen dataclass, and Lightning moves whatever a
    ``predict_step`` returns to CPU by rebuilding dataclasses through
    ``__setattr__`` -- which a frozen one refuses. So the class and its fields
    travel separately and are put back together at reassembly.

    :param loader: The eval loader the batch came from, for its column names.
    :param step_data: The batch that was predicted.
    :param leaf: The group's ``group_by`` value tuple.
    :param preds: What the inference method returned.
    """
    cont_keys = (*loader.cond_cont_keys, *loader.resp_keys)
    return {
        "preds_cls": type(preds),
        "preds_fields": {f.name: getattr(preds, f.name) for f in fields(preds)},
        "obs": _pred_obs_from_leaf(loader.group_cols, leaf, preds),
        "obsm": _get_pred_obsm_dict(step_data, preds, cont_keys),
    }


def predictions_to_adata(
    data_dims: DataDimensions,
    records: list[dict[str, Any]],
    *,
    return_raw: bool = False,
) -> AnnData | tuple[AnnData, PredictionData]:
    """Reassembles one ``AnnData`` from what a prediction run produced.

    :param data_dims: The dimensionalities, for the output ``var`` index.
    :param records: One ``{"preds_cls", "preds_fields", "obs", "obsm"}`` per
        predicted group, as
        :meth:`~sckitflow.trainer.TrainingPlan.predict_step` returns them.
    :param return_raw: Also return the concatenated raw `PredictionData`.
    """
    if not records:
        return _predict_empty(data_dims, return_raw)

    all_obsm: dict[str, list[np.ndarray]] = defaultdict(list)
    for record in records:
        for key, val in record["obsm"].items():
            all_obsm[key].append(val)

    return _aggregate_nodes_pred(
        data_dims,
        # rebuild the frozen `PredictionData` the predict step had to take apart
        [r["preds_cls"](**r["preds_fields"]) for r in records],
        [r["obs"] for r in records],
        all_obsm,
        return_raw=return_raw,
    )


@torch.inference_mode()
def predict_adata(
    datamodule: FlowDataModule,
    inference_method: SupportsInference,
    adata: AnnData,
    *,
    return_raw: bool = False,
    max_per_group: int | None = None,
    require_target_state: bool = True,
    control_values_dict: dict[str, str] | None = None,
    matched_keys: Mapping[tuple, tuple] | None = None,
    control_adata: AnnData | None = None,
    predict_kwargs: dict[str, Any] | None = None,
    seed: int = 0,
) -> AnnData | tuple[AnnData, PredictionData]:
    """Predicts over ``adata``, one deterministic pass per group, into one `AnnData`.

    A plain pass rather than ``Trainer.predict``: predicting needs no loop
    machinery, and our eval loader cannot shard, so multi-device prediction would
    silently duplicate rows. For a prediction writer or multi-device, drive
    `~sckitflow.trainer.TrainingPlan.predict_step` through a `lightning.Trainer`
    instead.

    :param datamodule: Supplies the schema and the per-group eval loader.
    :param inference_method: The method whose ``predict`` is called per group.
    :param adata: The input adata carrying the metadata to predict over.
    :param return_raw: If ``True``, also return the concatenated `PredictionData`.
    :param max_per_group: Per-group cap on observations.
    :param require_target_state: Whether ``adata`` must carry a target state representation.
    :param control_values_dict: Mapping from each condition level to its control value.
    :param matched_keys: ``{source group key: target group key}`` pairs for fixed matching.
    :param control_adata: Optional separate control (source) pool.
    :param predict_kwargs: Forwarded to the inference method's ``predict``.
    :param seed: Each group's noise comes from generators derived from it and the group's position.
    """
    loader = datamodule.set_predict_data(
        adata,
        max_per_group=max_per_group,
        require_target_state=require_target_state,
        control_values_dict=control_values_dict,
        matched_keys=matched_keys,
        control_adata=control_adata,
    ).predict_dataloader()
    predict_kwargs = {} if predict_kwargs is None else predict_kwargs

    was_training = inference_method.module.training
    inference_method.module.eval()
    records = []
    try:
        for i, (step_data, leaf) in enumerate(tqdm(loader, total=len(loader), desc="Predicting")):
            reference = next((t for t in (step_data["target_state"], step_data["source_state"]) if t is not None), None)
            generator, _ = generators(seed, i, device=None if reference is None else reference.device)
            preds = inference_method.predict(step_data, generator=generator, **predict_kwargs)
            records.append(prediction_record(loader, step_data, leaf, preds))
    finally:
        inference_method.module.train(was_training)

    return predictions_to_adata(datamodule.data_dims, records, return_raw=return_raw)
