from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING, TypedDict, Unpack

import numpy as np
import pandas as pd
from anndata import AnnData

if TYPE_CHECKING:
    import torch

    from sckitflow.data._loader import EvalLoader, Loader

from sckitflow._types import TargetCovariatesEncodingId
from sckitflow.data._dims_registry import DataDimensionalitiesRegistry
from sckitflow.data._group_encoders import GroupEncoder, GroupEncoderId
from sckitflow.data.containers._categorical import CategoricalData
from sckitflow.data.containers._coupling import CouplingData
from sckitflow.data.containers._distribution import DistributionData
from sckitflow.data.containers._mixed_type import MixedTypeData
from sckitflow.data.containers._state import StateData
from sckitflow.data.schemas import (
    ConditionDataSchema,
    CouplingDataSchema,
    GroupsDataSchema,
    ResponseDataSchema,
    StateDataSchema,
)

__all__ = ["DataManagerKwargs", "LoaderKwargs", "DataManager"]


class LoaderKwargs(TypedDict, total=False):
    """The :class:`~sckitflow.data._loader.Loader` options :meth:`DataManager.get_dataloaders` forwards.

    Declared here rather than beside :class:`~sckitflow.data._loader.Loader` so that typing a call site
    does not import the scfit/annbatch stack (~2s) that the loader module pulls in.

    Everything describing *which* cells a loader streams (``primary_weights``, ``control_adata``,
    ``control_weights``) is derived by :meth:`DataManager.get_dataloaders` and so is deliberately absent:
    these are the knobs that survive being passed through.
    """

    to: str | None
    """scfit's batch backend. ``None`` (the default) keeps annbatch's native arrays -- numpy on the host,
    cupy on a GPU-resident read window -- which the loader then maps onto torch without copying."""

    dtype: torch.dtype | None
    """Torch dtype every emitted tensor is cast to. ``None`` leaves the streamed dtype alone."""

    device: str | None
    """Device the streamed batches must *already* be on -- an assertion, not a move (a mismatch raises
    rather than copying every batch). ``None`` skips the check."""

    seed: int
    """Seed for scfit's sampling schedule."""

    n_iters: int | None
    """Batches one pass yields. ``None`` = one epoch over the primary."""

    batch_size: int
    """Rows per emitted batch."""

    chunk_size: int
    """annbatch read-slice size; ``1`` reads per row, ``>1`` requires leaf-contiguous runs."""

    preload_nchunks: int | None
    """Chunks per annbatch read window. ``None`` = ``batch_size // chunk_size``."""


class DataManagerKwargs(TypedDict, total=False):
    """Keyword arguments accepted by :class:`DataManager`."""

    sample_rep: str | None
    """String identifier for the state representation. When provided, it should appear as key in
    `.obsm` attribute of annotated data objects. Otherwise, the representation will fall back to
    `.X`. Defaults to `None`."""

    conditions: dict[str, Collection[str]] | None
    """Mapping from each condition level to the corresponding columns, used to initialize the
    conditioning data schema. Defaults to `None`."""

    conditions_reps: dict[str, str] | None
    """Mapping from each condition level to the corresponding representation, used to initialize the
    conditioning data schema. Defaults to `None`."""

    conditions_covariates: Collection[str] | None
    """Collection of continuous condition covariates, used to initialize the conditioning data
    schema. Defaults to `None`."""

    control_values_dict: dict[str, str] | None
    """Dictionary mapping each condition level to the corresponding value used to indicate control
    observations. Defaults to `None`."""

    condition_state_key: str | None
    """The key for the continuous condition covariates to be viewed as state when
    `view_on_condition_space` is `True`. This argument is ignored otherwise. Defaults to `None`."""

    target_categorical_covs_dict: Mapping[str, TargetCovariatesEncodingId] | None
    """Mapping indicating the encoding used to transform categorical target covariates, used to
    initialize the target data schema. Defaults to `None`."""

    target_continuous_covs: Collection[str] | None
    """Collection of string identifiers for the continuous target covariates, used to initialize the
    target data schema. Defaults to `None`."""

    groups: Collection[str] | None
    """Collection of string identifiers for grouping columns, used to initialize the grouping data
    schema. Defaults to `None`."""

    groups_reps: dict[str, str] | None
    """Mapping for pre-computed representations of grouping covariates, used to initialize the target
    data schema. Defaults to `None`."""

    groups_encoding: dict[str, GroupEncoder | GroupEncoderId] | None
    """Mapping from each group column to a :class:`~sckitflow.data._group_encoders.GroupEncoder`
    (e.g. ``OneHot()``, ``Label()``, ``Affine(scale=2.0)``), used to initialize the grouping data
    schema. Encoders are serializable dataclasses that build their fitted transformer on demand. The
    string ids ``"label"`` / ``"one-hot"`` are accepted as shorthand for the parameter-free encoders.
    Defaults to `None`."""

    n_shared_dims: int | None
    """The number of shared dimensions to be considered when matching distributions over
    incomparable spaces, used to initialize the coupling data schema. Defaults to `None`."""

    source_rep: str | None
    """String identifier for the state representation of source states, used when matching
    distributions over incomparable spaces. Used to initialize the coupling data schema.
    Defaults to `None`."""


class DataManager:
    """Class for managing data configurations."""

    def __init__(self, **kwargs: Unpack[DataManagerKwargs]) -> None:
        """Initializes the object. See :class:`DataManagerKwargs` for parameter descriptions."""
        self._control_values_dict = kwargs.get("control_values_dict")
        self._condition_state_key = kwargs.get("condition_state_key")

        self._state_data_schema = StateDataSchema(sample_rep=kwargs.get("sample_rep"))
        self._condition_data_schema = ConditionDataSchema(
            conditions=kwargs.get("conditions"),
            conditions_reps=kwargs.get("conditions_reps"),
            conditions_covariates=kwargs.get("conditions_covariates"),
        )
        self._coupling_data_schema = CouplingDataSchema(
            source_rep=kwargs.get("source_rep"),
            target_rep=kwargs.get("sample_rep"),
            n_shared_dims=kwargs.get("n_shared_dims"),
        )
        self._target_data_schema = ResponseDataSchema(
            categorical_covs_dict=kwargs.get("target_categorical_covs_dict"),
            continuous_covs=kwargs.get("target_continuous_covs"),
        )
        self._groups_data_schema = GroupsDataSchema(
            groups=kwargs.get("groups"),
            groups_reps=kwargs.get("groups_reps"),
            groups_encoding=kwargs.get("groups_encoding"),
        )

    @property
    def _view_on_condition_space(self) -> bool:
        return self._condition_state_key is not None

    def _get_feature_names(
        self,
        adata: AnnData,
        n_features: int,
    ) -> pd.Index:
        """Determines feature names based on the state representation used."""
        if self._view_on_condition_space:
            sample_rep = self._condition_state_key
        else:
            sample_rep = self._state_data_schema.sample_rep

        # If we have a valid base name, build feature names as "base_0", "base_1", ...
        if sample_rep is not None:
            return pd.Index([f"{sample_rep}_{i}" for i in range(n_features)])

        # Fallback: use original var_names only if the number of features matches.
        if n_features == adata.shape[-1]:
            return adata.var_names

        # Last resort: generic feature names.
        return pd.Index([f"feature_{i}" for i in range(n_features)])

    def _get_state_data(
        self,
        adata: AnnData,
    ) -> StateData:
        return self._state_data_schema.get_data(adata)

    def _get_condition_data(
        self,
        adata: AnnData,
    ) -> MixedTypeData | None:
        return self._condition_data_schema.get_data(adata)

    def _get_coupling_data(self, adata: AnnData) -> tuple[CouplingData, CouplingData]:
        return self._coupling_data_schema.get_data(adata)

    def _get_groups_data(
        self,
        adata: AnnData,
    ) -> CategoricalData:
        return self._groups_data_schema.get_data(adata)

    def _get_target_data(
        self,
        adata: AnnData,
    ) -> MixedTypeData | None:
        return self._target_data_schema.get_data(adata)

    def _get_distribution_data(
        self,
        adata: AnnData,
        require_target_state: bool = True,
    ) -> DistributionData:
        state_data: StateData | None = self._get_state_data(adata) if require_target_state else None
        condition_data: MixedTypeData = self._get_condition_data(adata)
        response_data: MixedTypeData = self._get_target_data(adata)
        groups_data: CategoricalData = self._get_groups_data(adata)
        if require_target_state:
            source_coupling_data, target_coupling_data = self._get_coupling_data(adata)
        else:
            # coupling reps default to the state representation (`.X`) unless explicitly
            # configured otherwise, so they are unavailable in the same cases as state_data.
            source_coupling_data, target_coupling_data = None, None

        distribution_data = DistributionData(
            state_data,
            response_data=response_data,
            condition_data=condition_data,
            groups_data=groups_data,
            source_coupling_data=source_coupling_data,
            target_coupling_data=target_coupling_data,
        )

        if self._view_on_condition_space:
            return distribution_data.view_on_condition_space(self._condition_state_key)
        return distribution_data

    def _get_data_dimensionalities(
        self,
        data: DistributionData,
        feature_names: pd.Index,
    ) -> DataDimensionalitiesRegistry:
        return DataDimensionalitiesRegistry.init_from_distribution_data(data, feature_names)

    def get_distribution_data(
        self,
        adata: AnnData,
        require_target_state: bool = True,
    ) -> DistributionData:
        """Compiles an annotated data object into a distribution data container.

        :param adata: The annotated data object to compile.
        :type adata: class: `AnnData`

        :param require_target_state: Whether the state representation (`.X` or the configured
            `obsm` sample representation) is required from `adata`. Set to `False` to compile
            only the distribution metadata (condition/group covariates) when no target state
            data is available, e.g. for prediction without target states. Defaults to `True`.
        :type require_target_state: class: `bool`
        """
        return self._get_distribution_data(
            adata,
            require_target_state=require_target_state,
        )

    def get_data_dimensionalities(
        self,
        adata: AnnData,
    ) -> DataDimensionalitiesRegistry:
        """Registers the data dimensionalities from the input data according to the current schema.

        :param adata: The annotated data object which to extract the dimensionalities from.
        :type adata: class: `AnnData`
        """
        data: DistributionData = self._get_distribution_data(adata)
        n_features = data.state_data.X.shape[1]
        feature_names = self._get_feature_names(adata, n_features)
        return self._get_data_dimensionalities(data, feature_names)

    def get_feature_names(self, adata: AnnData) -> pd.Index:
        """Registers the feature names from the input data according to the current schema.

        :param adata: The annotated data object which to extract the feature names from.
        :type adata: class: `AnnData`
        """
        n_features = self._get_state_data(adata).X.shape[1]
        return self._get_feature_names(adata, n_features)

    def get_dataloaders(
        self,
        adata: AnnData,
        *,
        split_by: str = "split",
        control_adata: AnnData | None = None,
        **loader_kwargs: Unpack[LoaderKwargs],
    ) -> dict[str, Loader]:
        """Build one streaming data loader per split value of ``adata.obs[split_by]``.

        Selection is by scfit weights over scfit's own leaf factorization -- no subset copying. The whole
        ``adata`` is streamed; each split's primary weights pick out that split's perturbed groups
        (controls get weight 0). Controls are shared across splits: pass ``control_adata`` (a separate
        pool -- faster to load, and the cross-dataset case), or, when omitted, they are drawn from
        ``adata`` itself by weight.

        :param adata: The annotated data object; must carry the ``split_by`` column in ``.obs``.
        :type adata: class: `AnnData`

        :param split_by: The ``.obs`` column whose values define the splits. Defaults to ``"split"``.
        :type split_by: class: `str`

        :param control_adata: Optional separate control (source) pool, shared by every split.
        :type control_adata: class: `AnnData | None`

        :param loader_kwargs: Forwarded to each :class:`Loader`. See :class:`LoaderKwargs` for the
            accepted options.

        :returns: Mapping from each split value (that has perturbed groups) to its loader.
        :rtype: class: `dict[str, Loader]`
        """
        from sckitflow.data._loader import Loader

        if split_by not in adata.obs.columns:
            raise KeyError(f"{split_by!r} not found in adata.obs (columns: {list(adata.obs.columns)}).")

        # One O(N) pass to the unique (split, group) combinations -- category-level, no per-cell logic.
        group_cols = (*tuple(self._groups_data_schema.groups), *self._condition_data_schema.all_condition_cols)
        uniq = adata.obs.loc[:, [split_by, *group_cols]].drop_duplicates()
        combos = pd.Series(list(map(tuple, uniq[list(group_cols)].to_numpy())), index=uniq.index)
        splits = uniq[split_by].astype(str)

        # A group is control iff every control condition column holds its control value.
        conditions = self._condition_data_schema.conditions
        checks = [
            (col, str(val)) for level, val in (self._control_values_dict or {}).items() for col in conditions[level]
        ]
        is_control = (
            np.logical_and.reduce([uniq[col].astype(str).to_numpy() == val for col, val in checks])
            if checks
            else np.zeros(len(uniq), dtype=bool)
        )

        control_weights = dict.fromkeys(combos[is_control], 1.0)
        return {
            split_value: Loader(
                adata,
                dm=self,
                primary_weights=dict.fromkeys(group_combos, 1.0),
                control_adata=control_adata,
                control_weights=None if control_adata is not None else (control_weights or None),
                **loader_kwargs,
            )
            for split_value, group_combos in combos[~is_control].groupby(splits[~is_control], observed=True)
        }

    def get_eval_loader(
        self,
        adata: AnnData,
        *,
        max_per_group: int | None = None,
        require_target_state: bool = True,
        control_adata: AnnData | None = None,
        control_values_dict: dict[str, str] | None = None,
        subsample: str = "head",
        to: str | None = None,
        dtype: torch.dtype | None = None,
        device: str | None = None,
        seed: int = 0,
    ) -> EvalLoader:
        """Build a deterministic, per-group :class:`~sckitflow.data._loader.EvalLoader` for prediction.

        Walks every perturbed group once (or a ``max_per_group`` cap), each matched to its control leaf,
        yielding ``(StepData, leaf)``. Controls are identified from ``control_values_dict`` (this argument
        overrides the instance's, to allow inference over arbitrary control keys); when it resolves to
        ``None`` (unpaired) every group is predicted with no source. Selection is by scfit weights over the
        whole ``adata`` -- no subset copying.

        :param adata: The annotated data object to predict over.
        :type adata: class: `AnnData`

        :param max_per_group: Per-group cap on cells: ``None`` = all, ``N`` = at most N, ``1`` =
            predict-once-per-condition (dedup). Defaults to ``None``.
        :type max_per_group: class: `int | None`

        :param require_target_state: Whether a target state representation is read from ``adata``. Set to
            ``False`` for metadata-only prediction (``StepData["target_state"]`` is ``None``). Defaults to ``True``.
        :type require_target_state: class: `bool`

        :param control_adata: Optional separate control (source) pool, matched on the group columns.
        :type control_adata: class: `AnnData | None`

        :param control_values_dict: Overrides the instance's control values for this call. Defaults to
            ``None`` (use the instance's).
        :type control_values_dict: class: `dict[str, str] | None`

        :param subsample: How ``max_per_group`` picks cells: ``"head"``, ``"random"``, or a callable.
        :type subsample: class: `str`

        :param dtype: Torch dtype every emitted tensor is cast to; ``None`` leaves them as streamed.
        :type dtype: class: `torch.dtype | None`

        :param device: Device the streamed batches must already be on; ``None`` skips the check. A
            mismatch raises rather than copying every batch.
        :type device: class: `str | None`
        """
        from sckitflow.data._loader import EvalLoader

        cvd = control_values_dict if control_values_dict is not None else self._control_values_dict
        group_cols = (*tuple(self._groups_data_schema.groups), *self._condition_data_schema.all_condition_cols)
        uniq = adata.obs.loc[:, list(group_cols)].drop_duplicates()
        combos = list(map(tuple, uniq.to_numpy()))

        # A group is control iff every control condition column holds its control value.
        conditions = self._condition_data_schema.conditions
        checks = [(col, str(val)) for level, val in (cvd or {}).items() for col in conditions[level]]
        is_control = (
            np.logical_and.reduce([uniq[col].astype(str).to_numpy() == val for col, val in checks])
            if checks
            else np.zeros(len(uniq), dtype=bool)
        )
        # Primary = perturbed groups (all groups when unpaired); control link = the control groups.
        primary_weights = {c: 1.0 for c, ctrl in zip(combos, is_control, strict=True) if not ctrl}
        control_weights = {c: 1.0 for c, ctrl in zip(combos, is_control, strict=True) if ctrl}
        return EvalLoader(
            adata,
            dm=self,
            primary_weights=primary_weights,
            control_adata=control_adata,
            control_weights=control_weights,
            require_target_state=require_target_state,
            max_per_group=max_per_group,
            subsample=subsample,
            to=to,
            dtype=dtype,
            device=device,
            seed=seed,
        )

    @property
    def control_values_dict(self) -> dict[str, str] | None:
        """Exposes the homonymous attribute set at initialization."""
        return self._control_values_dict

    @property
    def state_data_schema(self) -> StateDataSchema:
        """Exposes the state data schema."""
        return self._state_data_schema

    @property
    def condition_data_schema(self) -> ConditionDataSchema:
        """Exposes the condition data schema."""
        return self._condition_data_schema

    @property
    def coupling_data_schema(self) -> CouplingDataSchema:
        """Exposes the coupling data schema."""
        return self._coupling_data_schema

    @property
    def groups_data_schema(self) -> GroupsDataSchema:
        """Exposes the coupling data schema."""
        return self._groups_data_schema

    @property
    def target_data_schema(self) -> ResponseDataSchema:
        """Exposes the target data schema."""
        return self._target_data_schema

    @property
    def condition_state_key(self) -> str | None:
        """Returns the key used to define the condition covariates to be viewed as state."""
        return self._condition_state_key
