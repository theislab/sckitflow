"""Turning a run of predictions back into an ``AnnData``.

Free functions, not methods: reassembly is a pure transform of
``(predictions, dimensionalities)`` and holds no state of its own.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import fields
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from tqdm import tqdm

from sckitflow._random import PREDICT, generators
from sckitflow.core._types import PredictionData, StepData, concatenate_predictions
from sckitflow.core.methods._base import SupportsInference
from sckitflow.data._datamodule import FlowDataModule
from sckitflow.data._dims import DataDimensions
from sckitflow.data._utils import AnnDataLike

__all__ = ["prediction_record", "predictions_to_adata", "predict_adata"]

SAMPLE_COL = "sample_idx"
SOURCE_COL = "source_obs_name"


class _RestoreTrainingMode:
    """Context manager that puts a module into eval mode and restores its prior state on exit.

    Encapsulates the "always restore, even on exception" contract inside a
    ``with`` block, so call sites carry no ``try``. ``__exit__`` returns
    ``False`` so exceptions raised inside the block still propagate.
    """

    def __init__(self, module: torch.nn.Module) -> None:
        self._module = module
        self._was_training = module.training

    def __enter__(self) -> _RestoreTrainingMode:
        self._module.eval()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Literal[False]:
        self._module.train(self._was_training)
        return False


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
    # An unconditional run (no group columns) still needs one row per observation,
    # but the loader may still carry a sentinel leaf like ('g',): ignore it and
    # let the index size the frame. When group columns exist, require an exact match.
    if group_cols:
        if len(leaf) != len(group_cols):
            raise ValueError(
                f"leaf has {len(leaf)} values but group_cols has {len(group_cols)}: {leaf!r} vs {group_cols!r}"
            )
        cols = {col: np.repeat(val, n_pred_obs) for col, val in zip(group_cols, leaf, strict=False)}
    else:
        cols = {}
    return pd.DataFrame(cols, index=pd.RangeIndex(n_pred_obs))


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
    *,
    return_raw: bool = False,
    aggregate: Literal["samples", "mean"] = "samples",
) -> AnnData | tuple[AnnData, PredictionData]:
    merged_pred = concatenate_predictions(all_preds)

    X_np = _to_numpy(merged_pred.X)

    obs_final = pd.concat(all_obs, axis=0, ignore_index=True)

    obsm_final = {k: np.concatenate(v, axis=0) for k, v in all_obsm.items()}

    samples = obsm_final.get("raw_samples")
    if aggregate == "samples" and samples is not None and samples.ndim == 3:
        # one row per (input row, sample), row-major: each input row's samples are adjacent
        del obsm_final["raw_samples"]
        n_obs, n_samples = samples.shape[:2]
        X_np = samples.reshape(n_obs * n_samples, -1)
        obs_final = obs_final.iloc[np.repeat(np.arange(n_obs), n_samples)].reset_index(drop=True)
        obs_final[SAMPLE_COL] = np.tile(np.arange(n_samples), n_obs)
        for key, val in obsm_final.items():
            if key == "trajectory" and val.ndim == 4:  # (B, T, N, D) -> (B * N, T, D)
                obsm_final[key] = val.transpose(0, 2, 1, 3).reshape(n_obs * n_samples, val.shape[1], val.shape[3])
            else:
                obsm_final[key] = np.repeat(val, n_samples, axis=0)
    elif aggregate not in ("samples", "mean"):
        raise ValueError(f"`aggregate` must be 'samples' or 'mean', got {aggregate!r}.")
    obs_final.index = obs_final.index.astype(str)

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


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------


def _group_size(step_data: dict) -> int:
    """Number of observations in one group's ``step_data``.

    Reads it off any per-observation tensor, in order of preference:
    ``target_state``, ``source_state``, then the ``*_condition_data`` /
    ``*_response_data`` dicts. All these are obs-first, as consumed elsewhere
    in this module.
    """
    for key in ("target_state", "source_state"):
        v = step_data.get(key)
        if v is not None and isinstance(v, torch.Tensor):
            return int(v.shape[0])
    for key in (
        "target_condition_data",
        "target_response_data",
        "source_condition_data",
        "source_response_data",
    ):
        data = step_data.get(key)
        if not data:
            continue
        for v in data.values():
            if isinstance(v, torch.Tensor):
                return int(v.shape[0])
    raise ValueError("Cannot infer group size from step_data: no per-observation tensor found.")


def _slice_step_data(step_data: dict, start: int, end: int) -> dict:
    """Slice every per-observation tensor in ``step_data`` to ``[start:end]``.

    All tensors in ``step_data`` are obs-first (see :func:`_group_size`), so
    the slice is applied on dim 0 and recursed into nested dicts. ``None`` and
    non-tensor values are passed through unchanged.
    """
    out: dict = {}
    for k, v in step_data.items():
        if v is None:
            out[k] = None
        elif isinstance(v, dict):
            out[k] = _slice_step_data(v, start, end)
        elif isinstance(v, torch.Tensor):
            out[k] = v[start:end]
        else:
            out[k] = v
    return out


def _obs_dim_for_field(field_name: str, ndim: int) -> int:
    """Which dim of a ``PredictionData`` field holds ``n_obs``.

    Matches the layout conventions of :func:`_get_pred_traj` and
    :func:`_get_pred_raw_samples`, so chunked and unchunked runs agree on how
    the outputs are interpreted downstream.
    """
    if field_name == "X":
        return 0
    if field_name == "raw_samples":
        # (n_obs, D) or (n_samples, n_obs, D)
        return 0 if ndim == 2 else 1
    if field_name == "traj":
        # (n_obs, D) / (T, n_obs, D) / (I, T, n_obs, D)
        return {2: 0, 3: 1, 4: 2}.get(ndim, 0)
    # Unknown field: assume obs-first.
    return 0


def _concat_chunks(preds_list: list[PredictionData], chunk_sizes: list[int]) -> PredictionData:
    """Concatenate per-chunk predictions along the observation dim.

    ``chunk_sizes`` are the number of observations that each chunk was asked
    to predict. The obs dim of each ``PredictionData`` field is located via
    :func:`_obs_dim_for_field` and checked against the expected cumulative /
    next-chunk size before concatenating, so shape regressions raise instead
    of silently producing an ill-shaped result.
    """
    if len(preds_list) == 1:
        return preds_list[0]
    if len(preds_list) != len(chunk_sizes):
        raise ValueError(f"{len(preds_list)} predictions but {len(chunk_sizes)} chunk sizes")

    fields_out: dict[str, Any] = {}
    for f in fields(preds_list[0]):
        acc = getattr(preds_list[0], f.name)
        cum_size = chunk_sizes[0]
        for i in range(1, len(preds_list)):
            new_val = getattr(preds_list[i], f.name)
            if acc is None and new_val is None:
                continue
            if acc is None or new_val is None:
                raise ValueError(
                    f"field {f.name!r}: present in some chunks but not all (chunk {i} = {type(new_val).__name__})"
                )
            if not isinstance(acc, torch.Tensor) or not isinstance(new_val, torch.Tensor):
                # Non-tensor field: assume it's identical across chunks.
                continue

            dim = _obs_dim_for_field(f.name, acc.ndim)
            if acc.shape[dim] != cum_size or new_val.shape[dim] != chunk_sizes[i]:
                raise ValueError(
                    f"field {f.name!r}: expected obs dim {dim} to be "
                    f"({cum_size}, {chunk_sizes[i]}) but got "
                    f"({acc.shape[dim]}, {new_val.shape[dim]}); "
                    f"shapes are {tuple(acc.shape)} and {tuple(new_val.shape)}"
                )
            acc = torch.cat([acc, new_val], dim=dim)
            cum_size += chunk_sizes[i]
        fields_out[f.name] = acc
    return type(preds_list[0])(**fields_out)


def _chunks_for(n_obs: int, chunk_size: int | None) -> list[tuple[int, int]]:
    """Split ``n_obs`` observations into ``chunk_size``-sized ``(start, end)`` bounds."""
    if chunk_size is None or n_obs <= chunk_size:
        return [(0, n_obs)]
    return [(start, min(start + chunk_size, n_obs)) for start in range(0, n_obs, chunk_size)]


def _iter_chunks(loader: Any, chunk_size: int | None) -> Iterator[tuple[int, int, int, dict, int, int, tuple]]:
    """Yield one tuple per observation-chunk across every group in ``loader``.

    Each yield is ``(group_idx, chunk_idx, n_chunks_in_group, step_data, start,
    end, leaf)``. A group with ``n_obs <= chunk_size`` -- or any group when
    ``chunk_size`` is None -- yields a single ``(0, n_obs)`` chunk, so this
    reduces to ``enumerate(loader)`` with no chunking. Lazy: only one group's
    ``step_data`` is held at a time.
    """
    for group_idx, (step_data, leaf) in enumerate(loader):
        bounds = _chunks_for(_group_size(step_data), chunk_size)
        n_chunks = len(bounds)
        for chunk_idx, (start, end) in enumerate(bounds):
            yield group_idx, chunk_idx, n_chunks, step_data, start, end, leaf


# ---------------------------------------------------------------------------
# Prediction entry points
# ---------------------------------------------------------------------------


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
    obs = _pred_obs_from_leaf(loader.group_cols, leaf, preds)
    source_names = getattr(loader, "source_obs_names", {}).get(leaf)
    if source_names is not None:
        obs[SOURCE_COL] = source_names
    return {
        "preds_cls": type(preds),
        "preds_fields": {f.name: getattr(preds, f.name) for f in fields(preds)},
        "obs": obs,
        "obsm": _get_pred_obsm_dict(step_data, preds, cont_keys),
    }


def predictions_to_adata(
    data_dims: DataDimensions,
    records: list[dict[str, Any]],
    *,
    return_raw: bool = False,
    aggregate: Literal["samples", "mean"] = "samples",
) -> AnnData | tuple[AnnData, PredictionData]:
    """Reassembles one ``AnnData`` from what a prediction run produced.

    :param data_dims: The dimensionalities, for the output ``var`` index.
    :param records: One ``{"preds_cls", "preds_fields", "obs", "obsm"}`` per
        predicted group, as
        :meth:`~sckitflow.trainer.TrainingPlan.predict_step` returns them.
    :param return_raw: Also return the concatenated raw `PredictionData`.
    :param aggregate: How to lay out several samples per input row; see `predict_adata`.
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
        aggregate=aggregate,
    )


@torch.inference_mode()
def predict_adata(
    datamodule: FlowDataModule,
    inference_method: SupportsInference,
    adata: AnnDataLike,
    *,
    return_raw: bool = False,
    aggregate: Literal["samples", "mean"] = "samples",
    max_per_group: int | None = None,
    require_target_state: bool = True,
    control_values_dict: dict[str, str] | None = None,
    matched_keys: Mapping[tuple, tuple] | None = None,
    predict_kwargs: dict[str, Any] | None = None,
    seed: int = 0,
    chunk_size: int | None = None,
    progress_total: bool = True,
) -> AnnData | tuple[AnnData, PredictionData]:
    """Predicts over ``adata``, one deterministic pass per group, into one `AnnData`.

    A plain pass rather than ``Trainer.predict``: predicting needs no loop
    machinery, and our eval loader cannot shard, so multi-device prediction would
    silently duplicate rows. For a prediction writer or multi-device, drive
    `~sckitflow.trainer.TrainingPlan.predict_step` through a `lightning.Trainer`
    instead, passing ``dataloaders=datamodule.dm.get_eval_loader(adata, ...)``.

    :param datamodule: Supplies the schema and the streamed dtype.
    :param inference_method: The method whose ``predict`` is called per group.
    :param adata: The input adata carrying the metadata to predict over, or several read as one dataset.
    :param return_raw: If ``True``, also return the concatenated `PredictionData`.
    :param aggregate: With several samples per input row (``n_samples``), ``"samples"`` returns one row per
        (input row, sample), the row's obs repeated and the sample's index in ``obs["sample_idx"]``.
        ``"mean"`` returns one row per input row holding the mean over samples, with every sample in
        ``obsm["raw_samples"]``. With one sample per row both give the same output. When predictions flow
        from (or are conditioned on) source cells, ``obs["source_obs_name"]`` names each row's source cell.
    :param max_per_group: Per-group cap on observations, forwarded to the eval loader.
    :param require_target_state: Whether ``adata`` must carry a target state representation.
    :param control_values_dict: Mapping from each condition level to its control value.
    :param matched_keys: ``{source group key: target group key}`` pairs for fixed matching.
    :param predict_kwargs: Forwarded to the inference method's ``predict``.
    :param seed: Each group's noise comes from generators derived from it and the group's position,
        keyed as in :meth:`~sckitflow.trainer.TrainingPlan.predict_step`, so both predict alike.
    :param chunk_size: If set, split each group's observations into consecutive chunks of at most
        ``chunk_size`` and call ``predict`` once per chunk, then concatenate the chunk results back
        into the group's single `PredictionData`. This trades a bit of scheduling overhead for a
        much smaller forward-pass footprint (peak memory per call drops from ``O(n_obs)`` to
        ``O(chunk_size)``). The observation dimension is preserved exactly: the reassembled output
        has the same rows as with ``chunk_size=None``. Noise differs from a single-chunk run because
        each chunk draws from its own generator (keyed by chunk index); only the first chunk of a
        group shares the seed with the unchunked run. With ``chunk_size=None`` (default), behaviour
        is unchanged.
    :param progress_total: If ``True`` (default), pre-count the number of observation-chunks by
        iterating the loader once before the real pass, so the progress bar shows a proper
        ``x/y`` and ETA. The pre-count is skipped when the loader is a one-shot iterator
        (``iter(loader) is loader``), since walking it would consume the real pass; in that case
        the progress bar shows a running chunk count without a total (``total=None``). If
        ``False``, only one pass is made and the progress bar shows a running chunk count
        without a total.
    """
    if chunk_size is not None and chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    loader_args = {
        "to": None,
        "dtype": datamodule.dtype,
        "max_per_group": max_per_group,
        "require_target_state": require_target_state,
        "control_values_dict": control_values_dict,
        "matched_keys": matched_keys,
    }

    loader = datamodule.dm.get_eval_loader(adata, **loader_args)
    predict_kwargs = {} if predict_kwargs is None else predict_kwargs

    # Pre-count chunks so tqdm can render x/total and a real ETA. This is safe
    # only when the loader is re-iterable: containers hand out a fresh iterator
    # each walk, whereas an Iterator returns itself and would be consumed by
    # the count, leaving the real pass empty. Decide from the type up front
    # rather than catching the resulting empty-pass failure later.
    total_chunks: int | None = None
    if progress_total and not isinstance(loader, Iterator):
        total_chunks = sum(len(_chunks_for(_group_size(sd), chunk_size)) for sd, _ in loader)

    records: list[dict[str, Any]] = []
    with _RestoreTrainingMode(inference_method.module):
        current_group: int | None = None
        current_step_data: StepData | None = None
        current_leaf: tuple | None = None
        chunk_preds: list[PredictionData] = []
        chunk_sizes: list[int] = []

        bar_kwargs: dict[str, Any] = {"desc": "Predicting", "unit": "chunk", "total": total_chunks}
        if total_chunks is None:
            # Without a total, tqdm's default format renders "0chunk [00:00, ?chunk/s]"
            # on the first frame (no space, no percent). Use an explicit format so it
            # reads as "N chunks processed so far".
            bar_kwargs["bar_format"] = "{desc}: {n} chunks [{elapsed}, {rate_fmt}]{postfix}"

        bar = tqdm(_iter_chunks(loader, chunk_size), **bar_kwargs)

        for group_idx, chunk_idx, n_chunks, step_data, start, end, leaf in bar:
            bar.set_postfix(group=group_idx, chunk=f"{chunk_idx + 1}/{n_chunks}")

            if group_idx != current_group:
                if chunk_preds:
                    records.append(
                        prediction_record(
                            loader,
                            current_step_data,
                            current_leaf,
                            _concat_chunks(chunk_preds, chunk_sizes),
                        )
                    )
                chunk_preds = []
                chunk_sizes = []
                current_group = group_idx
                current_step_data = step_data
                current_leaf = leaf

            sub = _slice_step_data(step_data, start, end)
            reference = next(
                (t for t in (sub["target_state"], sub["source_state"]) if t is not None),
                None,
            )
            # The 3rd slot keys the noise stream; using it for the chunk index means
            # chunk 0 of group i is bit-identical to the single-chunk
            # (chunk_size=None) call for that group, while chunks 1..K draw
            # independent streams. This keeps chunked runs reproducible while
            # letting memory-bounded prediction progress at all.
            generator = generators(
                seed,
                PREDICT,
                chunk_idx,
                group_idx,
                device=None if reference is None else reference.device,
            )
            chunk_preds.append(inference_method.predict(sub, generator=generator, **predict_kwargs))
            chunk_sizes.append(end - start)

        if chunk_preds:
            records.append(
                prediction_record(
                    loader,
                    current_step_data,
                    current_leaf,
                    _concat_chunks(chunk_preds, chunk_sizes),
                )
            )

    return predictions_to_adata(
        datamodule.data_dims,
        records,
        return_raw=return_raw,
        aggregate=aggregate,
    )
