import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from tests.data.shared import with_split

from sckitflow.data._manager import DataManager
from sckitflow.data.containers._categorical import CategoricalData
from sckitflow.data.containers._coupling import CouplingData
from sckitflow.data.containers._distribution import DistributionData
from sckitflow.data.containers._mixed_type import MixedTypeData
from sckitflow.data.containers._state import StateData
from sckitflow.data.splitters import CombinationSplitter


def _make_manager(**overrides) -> DataManager:
    """DataManager with cell_line as group and drug as condition."""
    defaults = {
        "conditions": {"drug": ("drug",)},
        "conditions_reps": {"drug": "drug"},
        "groups": ("cell_line",),
        "groups_reps": {"cell_line": "cell_line"},
    }
    defaults.update(overrides)
    return DataManager(**defaults)


def _make_manager_with_continuous(**overrides) -> DataManager:
    """As :func:`_make_manager`, plus a continuous condition covariate (``X_repr``) from obsm."""
    return _make_manager(conditions_covariates=["X_repr"], **overrides)


class TestDistributionData:
    """get_distribution_data: building DistributionData from AnnData."""

    def test_types_and_lengths(self, adata_small: AnnData):
        manager = _make_manager()
        distr = manager.get_distribution_data(adata_small)

        assert isinstance(distr, DistributionData)
        assert len(distr) == adata_small.n_obs
        assert isinstance(distr.state_data, StateData)
        assert isinstance(distr.groups_data, CategoricalData)
        assert len(distr.state_data) == adata_small.n_obs
        assert len(distr.groups_data) == adata_small.n_obs
        if distr.condition_data is not None:
            assert isinstance(distr.condition_data, MixedTypeData)
            assert len(distr.condition_data) == adata_small.n_obs

    def test_slicing(self, adata_small: AnnData):
        manager = _make_manager()
        distr = manager.get_distribution_data(adata_small)
        n_slice = 5
        idxs = np.arange(n_slice)
        sliced = distr[idxs]

        assert len(sliced) == n_slice
        for attr in ("state_data", "groups_data", "condition_data", "response_data"):
            val = getattr(sliced, attr)
            if val is not None:
                assert len(val) == n_slice

    def test_empty_slice(self, adata_small: AnnData):
        manager = _make_manager()
        distr = manager.get_distribution_data(adata_small)
        sliced = distr[np.array([], dtype=int)]

        assert len(sliced) == 0
        for attr in ("state_data", "groups_data", "condition_data", "response_data"):
            val = getattr(sliced, attr)
            if val is not None:
                assert len(val) == 0

    def test_repr(self, adata_small: AnnData):
        manager = _make_manager()
        distr = manager.get_distribution_data(adata_small)
        r = repr(distr)
        assert "DistributionData" in r
        assert f"n_obs={adata_small.n_obs}" in r

    def test_coupling_defaults_to_state(self, adata_small: AnnData):
        manager = _make_manager()
        distr = manager.get_distribution_data(adata_small)
        assert isinstance(distr.source_coupling_data, CouplingData)
        assert isinstance(distr.target_coupling_data, CouplingData)
        assert len(distr.source_coupling_data) == adata_small.n_obs
        assert len(distr.target_coupling_data) == adata_small.n_obs


class TestRequireTargetState:
    """get_distribution_data with require_target_state=False (predict without target states)."""

    def test_get_distribution_data_state_is_none(self, adata_small: AnnData):
        manager = _make_manager()
        distr = manager.get_distribution_data(adata_small, require_target_state=False)

        assert distr.state_data is None
        assert len(distr) == adata_small.n_obs
        assert isinstance(distr.groups_data, CategoricalData)

    def test_get_distribution_data_works_without_x(self):
        """An AnnData with only `.obs` (no `.X`) can be compiled when require_target_state=False."""
        from tests.data.conftest import CELL_LINES, DRUGS

        obs = pd.DataFrame(
            {"cell_line": [CELL_LINES[0]] * 3, "drug": [DRUGS[0]] * 3},
        ).astype("category")
        ad = AnnData(obs=obs)
        # condition/group reps are looked up by value in `.uns`, regardless of `.X`
        ad.uns["drug"] = {DRUGS[0]: np.zeros((1, 4))}
        ad.uns["cell_line"] = {CELL_LINES[0]: np.zeros((1, 4))}

        manager = _make_manager()
        distr = manager.get_distribution_data(ad, require_target_state=False)

        assert distr.state_data is None
        assert len(distr) == 3


class TestConditionSpaceView:
    def test_get_distribution_data_with_condition_space(self, adata_small: AnnData):
        manager = _make_manager_with_continuous(condition_state_key="X_repr")
        # ensure X_repr exists in obsm
        if "X_repr" not in adata_small.obsm:
            adata_small.obsm["X_repr"] = np.random.randn(adata_small.n_obs, 10)

        distr = manager.get_distribution_data(adata_small)
        # state_data becomes the continuous covariate
        assert isinstance(distr.state_data, StateData)
        np.testing.assert_array_equal(distr.state_data.X, adata_small.obsm["X_repr"])

        # condition_data still exists (categorical part remains)
        assert distr.condition_data is not None
        # The continuous key "X_repr" should be removed. Since it was the only continuous key,
        # continuous_covariates becomes None. That's acceptable.
        if distr.condition_data.continuous_covariates is not None:
            assert "X_repr" not in distr.condition_data.continuous_covariates.mapping
        # categorical part remains
        assert distr.condition_data.categorical_covariates is not None

        # coupling data reinitialized
        assert distr.source_coupling_data is not None
        assert distr.target_coupling_data is not None
        assert len(distr.source_coupling_data) == len(distr.state_data)

    def test_get_distribution_data_invalid_condition_state_key(self, adata_small: AnnData):
        # Use a manager without continuous covariates -- only categorical conditions.
        manager = _make_manager(condition_state_key="invalid_key")
        with pytest.raises(KeyError, match="Key invalid_key not found"):
            manager.get_distribution_data(adata_small)

    def test_get_data_dimensionalities_with_condition_space(self, adata_small: AnnData):
        manager = _make_manager_with_continuous(condition_state_key="X_repr")
        if "X_repr" not in adata_small.obsm:
            adata_small.obsm["X_repr"] = np.random.randn(adata_small.n_obs, 10)

        dims = manager.get_data_dimensionalities(adata_small)
        # state dimension comes from continuous covariate
        assert dims.state_dim == adata_small.obsm["X_repr"].shape[1]
        # condition has a categorical part (drug) so categorical dim should be present
        assert dims.condition_reps_dims is not None and all(d > 0 for d in dims.condition_reps_dims.values())
        # the continuous covariate was consumed, so continuous condition dim should be 0 or None
        assert dims.condition_continuous_dims == {}


def _with_split(adata: AnnData) -> AnnData:
    """Attach a 'split' column: controls -> 'control', perturbed groups split train/val."""
    return with_split(adata, cols=("cell_line", "drug"), control_col="drug")


class TestGetDataloaders:
    """get_dataloaders: one streaming Loader per split, controls shared."""

    def test_one_loader_per_split(self, adata_small: AnnData):
        ad = _with_split(adata_small)
        dm = _make_manager(control_values_dict={"drug": "control"}, split_by="split")
        loaders = dm.get_dataloaders(ad, batch_size=8)

        # controls are the shared source, never their own split
        assert set(loaders) == {"train", "val"}
        for loader in loaders.values():
            assert len(loader) >= 1
            step_data = next(iter(loader))
            assert step_data["target_state"] is not None
            assert step_data["source_state"] is not None  # matched controls

    def test_missing_split_column_raises(self, adata_small: AnnData):
        """A declared split column that the data does not carry: the one case that must never pass."""
        dm = _make_manager(split_by="does_not_exist")
        with pytest.raises(KeyError, match="declared on this DataManager"):
            dm.get_dataloaders(adata_small)

    def test_a_split_never_streams_another_splits_cells(self, adata_small: AnnData):
        """A per-cell split makes every group span both splits; weights must still exclude the cells."""
        ad = adata_small.copy()
        # X row i is all-i, so a streamed row identifies itself; alternate splits within every group
        ad.X = np.repeat(np.arange(ad.n_obs, dtype=np.float32)[:, None], ad.n_vars, axis=1)
        ad.obs["split"] = pd.Categorical(["train" if i % 2 == 0 else "val" for i in range(ad.n_obs)])
        train_rows = set(np.flatnonzero((ad.obs["split"] == "train").to_numpy()).tolist())

        dm = _make_manager(split_by="split")  # no control values: every group is primary
        loaders = dm.get_dataloaders(ad, batch_size=2)
        assert set(loaders) == {"train", "val"}

        streamed = {int(v) for _ in range(10) for sd in loaders["val"] for v in sd["target_state"][:, 0].tolist()}
        assert streamed, "the val loader must stream something"
        assert not (streamed & train_rows), f"val loader streamed train cells: {sorted(streamed & train_rows)}"


class TestSplitOwnership:
    """The split is declared on the schema -- by a splitter or a column -- and checked, never assumed."""

    def test_splitter_and_split_by_together_are_rejected(self):
        with pytest.raises(ValueError, match="not both"):
            _make_manager(
                split_by="split",
                splitter=CombinationSplitter(group_keys=["cell_line", "drug"], always_train_keys=["cell_line"]),
            )

    def test_a_splitter_derives_the_split_without_touching_the_caller(self, adata_small: AnnData):
        """The schema owns the split, so no preprocessing step has to have written the column first."""
        dm = _make_manager(
            control_values_dict={"drug": "control"},
            splitter=CombinationSplitter(
                group_keys=["cell_line", "drug"],
                always_train_keys=["cell_line"],
                control_key="drug",
                test_fraction=0.5,
            ),
        )
        loaders = dm.get_dataloaders(adata_small, batch_size=8)

        assert set(loaders) == {"train", "test"}
        assert "split" not in adata_small.obs.columns  # derived on a shallow copy, not the caller's object

    def test_an_undeclared_split_column_is_refused(self, adata_small: AnnData):
        """Silently training on held-out cells is the one outcome the previous `split_by="split"` default hid."""
        ad = _with_split(adata_small)
        with pytest.raises(ValueError, match="declares neither"):
            _make_manager().get_dataloaders(ad, batch_size=8)

    def test_no_split_declared_and_none_present_trains_on_everything(self, adata_small: AnnData):
        loaders = _make_manager().get_dataloaders(adata_small, batch_size=8)
        assert set(loaders) == {"train"}


class TestControlPool:
    """A separate control pool is streamed as the source whether or not the schema declares pairing."""

    def test_a_paired_schema_streams_the_pool(self, adata_small: AnnData):
        dm = _make_manager(control_values_dict={"drug": "control"})
        step_data, _ = next(iter(dm.get_eval_loader(adata_small, control_adata=adata_small)))
        assert step_data["source_state"] is not None

    def test_an_unpaired_schema_streams_the_pool_too(self, adata_small: AnnData):
        """Pairing follows the data: a pool given at call time is a source, not something to discard."""
        step_data, _ = next(iter(_make_manager().get_eval_loader(adata_small, control_adata=adata_small)))
        assert step_data["source_state"] is not None

    def test_training_loaders_stream_the_pool_without_a_declared_pairing(self, adata_small: AnnData):
        loaders = _make_manager().get_dataloaders(adata_small, control_adata=adata_small, batch_size=8)
        assert next(iter(loaders["train"]))["source_state"] is not None


class TestGetEvalLoader:
    """get_eval_loader: deterministic per-group (StepData, leaf) for prediction."""

    def _n_groups(self, adata: AnnData, *, exclude_control: bool) -> int:
        obs = adata.obs[["cell_line", "drug"]].astype(str)
        if exclude_control:
            obs = obs[obs["drug"] != "control"]
        return obs.drop_duplicates().shape[0]

    def test_paired_predicts_noncontrol_matched_to_controls(self, adata_small: AnnData):
        dm = _make_manager(control_values_dict={"drug": "control"})
        el = dm.get_eval_loader(adata_small)

        assert len(el) == self._n_groups(adata_small, exclude_control=True)
        assert el.group_cols == ("cell_line", "drug")
        for step_data, leaf in el:
            assert step_data["target_state"] is not None
            assert step_data["source_state"] is not None  # matched controls
            assert "control" not in leaf  # only perturbed groups are predicted

    def test_max_per_group_one_dedups_and_reiterates(self, adata_small: AnnData):
        dm = _make_manager(control_values_dict={"drug": "control"})
        el = dm.get_eval_loader(adata_small, max_per_group=1)

        batches = list(el)
        assert all(step_data["target_state"].shape[0] == 1 for step_data, _ in batches)
        assert len({leaf for _, leaf in batches}) == len(batches)  # one per group
        assert len(list(el)) == len(el)  # re-iterable

    def test_unpaired_predicts_all_groups_without_source(self, adata_small: AnnData):
        dm = _make_manager()  # no control_values_dict
        el = dm.get_eval_loader(adata_small)

        assert len(el) == self._n_groups(adata_small, exclude_control=False)
        step_data, _ = next(iter(el))
        assert step_data["source_state"] is None  # unpaired: no control link

    def test_source_row_count_follows_the_target_not_the_control_pool(self, adata_small: AnnData):
        """Control pools rarely match a group's size; the emitted batch must still be internally aligned."""
        ad = adata_small.copy()
        # Thin the controls to one per cell_line (matching is on cell_line), so every group is matched to
        # a control pool strictly smaller than itself -- the source must be tiled up, not left short.
        is_ctrl = (ad.obs["drug"].astype(str) == "control").to_numpy()
        rank = ad.obs[is_ctrl].groupby("cell_line", observed=True).cumcount().reindex(ad.obs.index, fill_value=0)
        ad = ad[~is_ctrl | (rank.to_numpy() < 1)].copy()

        dm = _make_manager(control_values_dict={"drug": "control"})
        sizes = []
        for step_data, _ in dm.get_eval_loader(ad):
            n = step_data["target_state"].shape[0]
            sizes.append(n)
            assert step_data["source_state"].shape[0] == n
            for field in ("target_condition_data", "target_group_data"):
                assert all(v.shape[0] == n for v in (step_data[field] or {}).values())
        assert sizes and max(sizes) > 1, "groups must be bigger than their 1-cell control pool to be a real test"

    @pytest.mark.parametrize("cap", [1, 2, None])
    def test_metadata_only_batches_are_sized_from_the_group_not_the_controls(self, adata_small: AnnData, cap):
        """`reps=()` reads no primary cells, so the row count comes from the known (capped) group size."""
        dm = _make_manager(control_values_dict={"drug": "control"})
        el = dm.get_eval_loader(adata_small, require_target_state=False, max_per_group=cap)
        counts = {group: len(rows) for group, rows in adata_small.obs.groupby(["cell_line", "drug"], observed=True)}

        for step_data, leaf in el:
            assert step_data["target_state"] is None
            expected = counts[leaf] if cap is None else min(counts[leaf], cap)
            assert step_data["source_state"].shape[0] == expected
            assert all(v.shape[0] == expected for v in step_data["target_group_data"].values())

    def test_control_values_override_makes_it_paired(self, adata_small: AnnData):
        dm = _make_manager()  # instance is unpaired
        el = dm.get_eval_loader(adata_small, control_values_dict={"drug": "control"})

        assert len(el) == self._n_groups(adata_small, exclude_control=True)
        for step_data, leaf in el:
            assert "control" not in leaf
            assert step_data["source_state"] is not None


class TestMatchedKeys:
    """Fixed matching: `{source key: target key}` pairs instead of control-value matching."""

    # (cell_line, drug) pairs: a *non-control* source, which control_values_dict cannot express.
    PAIRS = {("HeLa", "aspirin"): ("HeLa", "ibuprofen"), ("Jurkat", "aspirin"): ("Jurkat", "paclitaxel")}

    def test_only_named_targets_stream_matched_to_their_named_source(self, adata_small: AnnData):
        dm = _make_manager(matched_keys=self.PAIRS)
        el = dm.get_eval_loader(adata_small)

        streamed = {leaf for _, leaf in el}
        assert streamed == set(self.PAIRS.values())  # nothing else is a target
        assert len(el) == len(self.PAIRS)
        for step_data, _ in el:
            assert step_data["source_state"] is not None

    @staticmethod
    def _tagged(adata: AnnData) -> tuple[AnnData, dict[tuple, int]]:
        """A copy whose ``.X[:, 0]`` holds each cell's (cell_line, drug) id, so a streamed row names its group."""
        ad = adata.copy()
        combos = list(ad.obs[["cell_line", "drug"]].astype(str).itertuples(index=False, name=None))
        ids = {combo: i for i, combo in enumerate(dict.fromkeys(combos))}
        ad.X[:, 0] = np.array([ids[c] for c in combos], dtype=np.float32)
        return ad, ids

    def test_the_source_is_the_paired_group_not_the_control(self, adata_small: AnnData):
        """Each target's source rows must be exactly its partner group's cells."""
        ad, ids = self._tagged(adata_small)
        dm = _make_manager(matched_keys=self.PAIRS)
        source_of = {target: source for source, target in self.PAIRS.items()}

        seen = 0
        for step_data, leaf in dm.get_eval_loader(ad):
            source_ids = {round(v) for v in step_data["source_state"][:, 0].tolist()}
            assert source_ids == {ids[source_of[leaf]]}, f"{leaf} flowed from the wrong group"
            seen += 1
        assert seen == len(self.PAIRS)

    def test_training_batches_bind_each_target_to_its_paired_source(self, adata_small: AnnData):
        """Same guarantee on the sampling path: the bound control sampler must follow the pair, not the group."""
        ad, ids = self._tagged(adata_small)
        ad = with_split(ad, cols=["cell_line", "drug"], labels=("train",))
        loaders = _make_manager(matched_keys=self.PAIRS, split_by="split").get_dataloaders(ad, batch_size=4)
        assert set(loaders) == {"train"}

        pair_of = {ids[target]: ids[source] for source, target in self.PAIRS.items()}
        batches = list(loaders["train"])
        assert batches, "the train loader must stream something"
        for step_data in batches:
            n = step_data["target_state"].shape[0]
            assert step_data["source_state"].shape[0] == n  # matched, row-aligned
            target_ids = {round(v) for v in step_data["target_state"][:, 0].tolist()}
            source_ids = {round(v) for v in step_data["source_state"][:, 0].tolist()}
            assert len(target_ids) == 1, "a batch is drawn from one group"
            assert source_ids == {pair_of[target_ids.pop()]}

    def test_the_callers_adata_is_never_written_to(self, adata_small: AnnData):
        """The pair column lives on a shallow copy; the caller's obs (and any view) stays untouched."""
        before = list(adata_small.obs.columns)
        dm = _make_manager(matched_keys=self.PAIRS)
        list(dm.get_eval_loader(adata_small))
        dm.get_dataloaders(adata_small, batch_size=4)

        assert list(adata_small.obs.columns) == before

    def test_matched_keys_override_at_inference(self, adata_small: AnnData):
        dm = _make_manager(matched_keys=self.PAIRS)  # registered pairs
        other = {("HEK293", "control"): ("HEK293", "aspirin")}
        el = dm.get_eval_loader(adata_small, matched_keys=other)

        assert {leaf for _, leaf in el} == set(other.values())

    def test_a_group_in_two_pairs_is_rejected(self, adata_small: AnnData):
        chain = {("HeLa", "control"): ("HeLa", "aspirin"), ("HeLa", "aspirin"): ("HeLa", "ibuprofen")}
        with pytest.raises(ValueError, match="more than one entry of matched_keys"):
            _make_manager(matched_keys=chain).get_eval_loader(adata_small)

    def test_an_unknown_key_is_rejected(self, adata_small: AnnData):
        with pytest.raises(KeyError, match="target keys not found"):
            _make_manager(matched_keys={("HeLa", "control"): ("HeLa", "nosuchdrug")}).get_eval_loader(adata_small)


class TestUnconditionalStreaming:
    """A schema with no groups/conditions still trains: one implicit group, no split."""

    def test_split_by_none_yields_a_single_train_loader(self, adata_small: AnnData):
        loaders = _make_manager().get_dataloaders(adata_small, batch_size=8)
        assert set(loaders) == {"train"}
        assert next(iter(loaders["train"]))["target_state"].shape[0] == 8

    def test_no_covariate_schema_trains_and_predicts(self, adata_small: AnnData):
        ad = adata_small.copy()
        ad.obs = ad.obs.iloc[:, :0]  # strip every covariate column
        dm = DataManager()

        loaders = dm.get_dataloaders(ad, batch_size=8)
        step_data = next(iter(loaders["train"]))
        assert step_data["target_state"].shape == (8, ad.n_vars)
        assert step_data["target_condition_data"] is None

        el = dm.get_eval_loader(ad)
        assert len(el) == 1  # a single implicit group
        pred_step, _ = next(iter(el))
        assert pred_step["target_state"].shape[0] == ad.n_obs
