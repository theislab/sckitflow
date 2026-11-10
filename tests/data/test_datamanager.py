import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scfit.data import open_adata
from tests.data.shared import with_split

from sckitflow.data._manager import DataManager, DataManagerConfig


def _make_manager(**overrides) -> DataManager:
    """DataManager with cell_line as group and drug as condition."""
    defaults = {
        "target_sample_rep": "X",
        "categorical_conditions": {"drug": ("drug",)},
        "categorical_reps": {"drug": "drug"},
        "groups": ("cell_line",),
        "groups_reps": {"cell_line": "cell_line"},
    }
    defaults.update(overrides)
    if (pairs := defaults.pop("matched_keys", None)) is not None:
        defaults["matched_pairs"] = tuple(pairs.items())
    return DataManager(DataManagerConfig(**defaults))


def _make_manager_with_continuous(**overrides) -> DataManager:
    """As :func:`_make_manager`, plus a continuous condition covariate (``X_repr``) from obsm."""
    return _make_manager(continuous_conditions=["obsm.X_repr"], **overrides)


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
    """The split is a declared column, checked, never assumed."""

    def test_an_undeclared_split_column_is_refused(self, adata_small: AnnData):
        """Silently training on held-out cells is the one outcome the previous `split_by="split"` default hid."""
        ad = _with_split(adata_small)
        with pytest.raises(ValueError, match="declares no"):
            _make_manager().get_dataloaders(ad, batch_size=8)

    def test_no_split_declared_and_none_present_trains_on_everything(self, adata_small: AnnData):
        loaders = _make_manager().get_dataloaders(adata_small, batch_size=8)
        assert set(loaders) == {"train"}


def _buffer(series: pd.Series) -> np.ndarray:
    """The array actually backing a column -- a categorical's codes, otherwise its values.

    ``Series.cat.codes`` builds a fresh array on each access, so it can never be compared for sharing;
    the internal ``_codes`` is the buffer a copy would have duplicated.
    """
    return series.array._codes if isinstance(series.dtype, pd.CategoricalDtype) else series.to_numpy()


class TestStreamingLeavesTheCallersDataAlone:
    """The invariant behind `with_derived_obs`: streaming derives obs columns without copying or mutating.

    The derived column -- the ``matched_keys`` pair id -- is a place sckitflow could have written to the
    caller's object.
    Nothing at runtime enforces this (``X`` and ``obsm`` are the caller's own arrays, and on a pandas
    without copy-on-write so are the obs columns), so it is pinned here instead.
    """

    def _managers(self):
        return {
            "matched_keys": _make_manager(matched_keys={("HeLa", "aspirin"): ("HeLa", "ibuprofen")}),
        }

    def _assert_shares_everything(self, streamed: AnnData, adata: AnnData) -> None:
        """The streamed adata must be a *shallow* derivative: same arrays, extra obs columns only."""
        assert streamed is not adata, "a derived column means a new object, not the caller's"
        assert streamed.X is adata.X
        for key in adata.obsm:
            assert streamed.obsm[key] is adata.obsm[key], f"obsm[{key!r}] was copied"
        for col in adata.obs.columns:
            assert np.shares_memory(_buffer(adata.obs[col]), _buffer(streamed.obs[col])), f"obs[{col!r}] was copied"

    @pytest.mark.parametrize("case", ["matched_keys"])
    def test_training_derives_obs_without_copying_or_mutating(self, adata_small: AnnData, case):
        adata_small.obsm["X_repr"] = np.zeros((adata_small.n_obs, 3), dtype=np.float32)
        before = list(adata_small.obs.columns)
        dm = self._managers()[case]

        loaders = dm.get_dataloaders(adata_small, batch_size=8)
        for loader in loaders.values():
            next(iter(loader))  # actually stream, so nothing is derived lazily on first read
            self._assert_shares_everything(loader._adatas[0], adata_small)
        assert list(adata_small.obs.columns) == before, "the caller's obs gained a column"

    @pytest.mark.parametrize("case", ["matched_keys"])
    def test_prediction_derives_obs_without_copying_or_mutating(self, adata_small: AnnData, case):
        adata_small.obsm["X_repr"] = np.zeros((adata_small.n_obs, 3), dtype=np.float32)
        before = list(adata_small.obs.columns)
        el = self._managers()[case].get_eval_loader(adata_small)

        next(iter(el))
        self._assert_shares_everything(el._adatas[0], adata_small)
        assert list(adata_small.obs.columns) == before


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
                assert all(v.shape[0] == 1 for v in (step_data[field] or {}).values())  # one row per group
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
            assert all(v.shape[0] == 1 for v in step_data["target_group_data"].values())  # one row per group

    def test_metadata_only_unpaired_batches_carry_their_row_count(self, adata_small: AnnData):
        """No state and no source: `n_obs` is the only row count, and per-observation responses match it."""
        ad = adata_small.copy()
        ad.obsm["resp"] = np.zeros((ad.n_obs, 2), dtype=np.float32)
        el = _make_manager(target_continuous_covs=("obsm.resp",)).get_eval_loader(ad, require_target_state=False)
        counts = {group: len(rows) for group, rows in ad.obs.groupby(["cell_line", "drug"], observed=True)}
        for step_data, leaf in el:
            assert step_data["source_state"] is None and step_data["target_state"] is None
            assert step_data["n_obs"] == counts[leaf] == step_data["target_response_data"]["resp"].shape[0]

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
        dm = DataManagerConfig(target_sample_rep="X").build()

        loaders = dm.get_dataloaders(ad, batch_size=8)
        step_data = next(iter(loaders["train"]))
        assert step_data["target_state"].shape == (8, ad.n_vars)
        assert step_data["target_condition_data"] is None

        el = dm.get_eval_loader(ad)
        assert len(el) == 1  # a single implicit group
        pred_step, _ = next(iter(el))
        assert pred_step["target_state"].shape[0] == ad.n_obs


class TestLinearFreeCoupling:
    """Quadratics without `rep_lin` (pure Gromov-Wasserstein) flow through to linear-free couplings."""

    def _with_quads(self, adata: AnnData) -> AnnData:
        rng = np.random.default_rng(0)
        adata.obsm["X_src_quad"] = rng.standard_normal((adata.n_obs, 4)).astype("float32")
        adata.obsm["X_tgt_quad"] = rng.standard_normal((adata.n_obs, 6)).astype("float32")
        return adata

    @pytest.mark.parametrize(
        "source_rep_quad, target_rep_quad",
        [(None, "obsm.X_tgt_quad"), ("obsm.X_src_quad", None)],
    )
    def test_config_requires_both_quads(self, source_rep_quad, target_rep_quad):
        with pytest.raises(ValueError, match="both `source_rep_quad` and `target_rep_quad`"):
            _make_manager(
                source_rep_quad=source_rep_quad,
                target_rep_quad=target_rep_quad,
            )


class TestTrainingOnlyCoupling:
    """The manager wires coupling into training loaders and withholds it from eval.

    `_loader_schema` is shared and does *not* carry `coupling_schema`; `_training_loader_schema` adds
    it. This is the only place the training/eval asymmetry is observable end to end.
    """

    def _with_coupling_reps(self, adata: AnnData) -> AnnData:
        rng = np.random.default_rng(0)
        adata.obsm["X_lin"] = rng.standard_normal((adata.n_obs, 5)).astype("float32")
        adata.obsm["X_src_quad"] = rng.standard_normal((adata.n_obs, 3)).astype("float32")
        adata.obsm["X_tgt_quad"] = rng.standard_normal((adata.n_obs, 4)).astype("float32")
        return adata

    def test_training_loaders_stream_coupling_no_split(self, adata_small: AnnData):
        """No-split branch: coupling reaches the loader when the schema declares it."""
        self._with_coupling_reps(adata_small)
        dm = _make_manager(
            control_values_dict={"drug": "control"},
            rep_lin="obsm.X_lin",
            source_rep_quad="obsm.X_src_quad",
            target_rep_quad="obsm.X_tgt_quad",
        )
        loaders = dm.get_dataloaders(adata_small, batch_size=8)
        sd = next(iter(loaders["train"]))
        assert sd["target_coupling_lin"].shape[1] == 5
        assert sd["source_coupling_lin"].shape[1] == 5
        assert sd["target_coupling_quad"].shape[1] == 4
        assert sd["source_coupling_quad"].shape[1] == 3

    def test_training_loaders_stream_coupling_with_a_split(self, adata_small: AnnData):
        """Split branch: same guarantee, on the path that actually reaches most users.

        Catches a `_loader_schema` / `_training_loader_schema` mix-up in the with-split return, which
        the no-split test above cannot see.
        """
        self._with_coupling_reps(adata_small)
        ad = _with_split(adata_small)
        dm = _make_manager(
            control_values_dict={"drug": "control"},
            split_by="split",
            rep_lin="obsm.X_lin",
            source_rep_quad="obsm.X_src_quad",
            target_rep_quad="obsm.X_tgt_quad",
        )
        loaders = dm.get_dataloaders(ad, batch_size=8)
        assert set(loaders) == {"train", "val"}
        sd = next(iter(loaders["train"]))
        assert sd["target_coupling_lin"].shape[1] == 5
        assert sd["target_coupling_quad"].shape[1] == 4
        assert sd["source_coupling_quad"].shape[1] == 3

    def test_eval_loader_emits_no_coupling_even_when_configured(self, adata_small: AnnData):
        """`EvalLoader` ignores the schema: coupling is training-only by design."""
        self._with_coupling_reps(adata_small)
        dm = _make_manager(
            control_values_dict={"drug": "control"},
            rep_lin="obsm.X_lin",
            source_rep_quad="obsm.X_src_quad",
            target_rep_quad="obsm.X_tgt_quad",
        )
        for sd, _leaf in dm.get_eval_loader(adata_small):
            assert sd["target_coupling_lin"] is None
            assert sd["target_coupling_quad"] is None
            assert sd["source_coupling_lin"] is None
            assert sd["source_coupling_quad"] is None

    def test_the_shared_schema_does_not_carry_coupling(self):
        """`_loader_schema` is coupling-free; only `_training_loader_schema` adds it, without replacement."""
        dm = _make_manager(rep_lin="obsm.X_lin")
        assert "coupling_schema" not in dm._loader_schema
        assert dm._training_loader_schema["coupling_schema"] is dm.coupling_data_schema
        # strict superset, and the shared keys survive identity-wise
        assert set(dm._loader_schema) < set(dm._training_loader_schema)
        for key, value in dm._loader_schema.items():
            assert dm._training_loader_schema[key] is value

    def test_pure_gw_streams_only_quads(self, adata_small: AnnData):
        """Without `rep_lin` the linear coupling keys stay None even in training."""
        self._with_coupling_reps(adata_small)
        dm = _make_manager(
            control_values_dict={"drug": "control"},
            source_rep_quad="obsm.X_src_quad",
            target_rep_quad="obsm.X_tgt_quad",
        )
        loaders = dm.get_dataloaders(adata_small, batch_size=8)
        sd = next(iter(loaders["train"]))
        assert sd["target_coupling_lin"] is None
        assert sd["source_coupling_lin"] is None
        assert sd["target_coupling_quad"].shape[1] == 4
        assert sd["source_coupling_quad"].shape[1] == 3

    def test_get_dataloaders_rejects_a_schema_owned_kwarg(self, adata_small: AnnData):
        """One owner per question: the schema owns `coupling_schema`, not the call."""
        dm = _make_manager(rep_lin="obsm.X_lin")
        with pytest.raises(TypeError, match="owned by the schema"):
            dm.get_dataloaders(adata_small, coupling_schema=dm.coupling_data_schema)


class TestSeveralAnnData:
    """Several AnnData are read as one dataset: the same groups and dims as the one they were cut from."""

    @staticmethod
    def _parts(adata: AnnData) -> list[AnnData]:
        """Two parts, each knowing only its own categories, so they must be unioned to be read together."""
        lines = adata.obs["cell_line"].cat.categories
        parts = [adata[adata.obs["cell_line"] == lines[0]].copy(), adata[adata.obs["cell_line"] != lines[0]].copy()]
        for part in parts:
            for col in part.obs.columns:
                part.obs[col] = part.obs[col].cat.remove_unused_categories()
        return parts

    def test_dims_and_groups_match_the_whole(self, adata_small: AnnData):
        dm = _make_manager(control_values_dict={"drug": "control"})
        parts = self._parts(adata_small)
        assert dm.get_data_dimensionalities(parts).feature_names.equals(
            dm.get_data_dimensionalities(adata_small).feature_names
        )
        assert set(dm.get_eval_loader(parts)._leaves) == set(dm.get_eval_loader(adata_small)._leaves)

    def test_training_streams_targets_with_controls_across_parts(self, adata_small: AnnData):
        dm = _make_manager(control_values_dict={"drug": "control"})
        batch = next(iter(dm.get_dataloaders(self._parts(adata_small), batch_size=8)["train"]))
        assert batch["source_state"] is not None and batch["target_state"].shape[0] == 8

    @pytest.mark.parametrize("holder", [0, 1])
    def test_a_table_may_live_in_one_part(self, adata_small: AnnData, holder: int):
        parts = self._parts(adata_small)
        parts[1 - holder].uns = {}
        dm = _make_manager(control_values_dict={"drug": "control"})
        assert dm.get_data_dimensionalities(parts).state_dim == dm.get_data_dimensionalities(adata_small).state_dim
        assert next(iter(dm.get_dataloaders(parts, batch_size=8)["train"]))["target_group_data"]["drug"] is not None
        assert parts[1 - holder].uns == {}  # the caller's AnnData gain no tables

    def test_an_entry_differing_between_parts_is_refused(self, adata_small: AnnData):
        parts = self._parts(adata_small)
        parts[1].uns["drug"] = {k: v + 1 for k, v in parts[1].uns["drug"].items()}
        with pytest.raises(ValueError, match=r"\.uns\['drug'\]\['\w+'\] differs between AnnData 0 and 1"):
            _make_manager().get_data_dimensionalities(parts)

    def test_a_value_without_an_entry_is_refused(self, adata_small: AnnData):
        parts = self._parts(adata_small)
        drug = next(iter(parts[0].uns["drug"]))
        for part in parts:
            part.uns["drug"] = {k: v for k, v in part.uns["drug"].items() if k != drug}
        with pytest.raises(KeyError, match=rf"\.uns\['drug'\] has no entry for \['{drug}'\]"):
            _make_manager().get_data_dimensionalities(parts)

    def test_parts_with_different_features_are_refused(self, adata_small: AnnData):
        parts = self._parts(adata_small)
        parts[1].var_names = [f"other_{i}" for i in range(parts[1].n_vars)]
        with pytest.raises(
            ValueError,
            match=r"AnnData 1 and AnnData 0 differ in their state \(`target_sample_rep`\): the same width but different feature names",
        ):
            _make_manager().get_data_dimensionalities(parts)


class TestGetRequiredFields:
    """get_required_fields: exactly the parts of an AnnData the loaders read."""

    def test_names_what_the_schema_reads(self):
        from anndata.acc import A

        dm = _make_manager_with_continuous(control_values_dict={"drug": "control"}, split_by="split")
        refs, uns = dm.get_required_fields()
        assert refs == [A.X, A.obsm["X_repr"], A.obs["cell_line"], A.obs["drug"], A.obs["split"]]
        assert uns == ["cell_line", "drug"]

    def test_the_named_parts_are_enough_to_train(self, adata_small: AnnData, tmp_path):
        ad = _with_split(adata_small)
        ad.obsm["X_repr"] = np.ones((ad.n_obs, 2), dtype=np.float32)
        ad.obsm["unused"], ad.obs["unused"] = np.zeros((ad.n_obs, 3)), "x"
        dm = _make_manager_with_continuous(control_values_dict={"drug": "control"}, split_by="split")
        ad.write_zarr(tmp_path / "d.zarr")
        refs, uns = dm.get_required_fields()
        reduced = open_adata(tmp_path / "d.zarr", *refs, uns=uns)
        assert "unused" not in reduced.obsm and "unused" not in reduced.obs
        loaders = dm.get_dataloaders(reduced, batch_size=8)
        assert set(loaders) == {"train", "val"}
        assert next(iter(loaders["train"]))["target_state"] is not None

    def test_a_state_only_schema_reads_only_its_state(self):
        from anndata.acc import A

        dm = DataManagerConfig(target_sample_rep="layers.counts").build()
        assert dm.get_required_fields() == ([A.layers["counts"]], [])

    def test_every_rep_role_is_named(self):
        from anndata.acc import A

        dm = DataManagerConfig(
            target_sample_rep="X",
            source_sample_rep="obsm.src",
            rep_lin="obsm.lin",
            source_rep_quad="obsm.sq",
            target_rep_quad="layers.tq",
            continuous_conditions=("obsm.dose",),
            target_continuous_covs=("layers.resp",),
        ).build()
        refs, _ = dm.get_required_fields()
        assert refs == [
            A.X,
            A.obsm["src"],
            A.obsm["lin"],
            A.obsm["sq"],
            A.layers["tq"],
            A.obsm["dose"],
            A.layers["resp"],
        ]
