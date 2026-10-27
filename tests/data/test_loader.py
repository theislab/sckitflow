"""Unit tests for :class:`sckitflow.data._loader.Loader`.

Covers the scfit-batch -> ``StepData`` bridge: weight-based group selection, the label-gid ->
cached categorical encoding (one row per group), continuous covariates riding as per-cell reps,
and the finite/cycling iteration contract the trainer relies on.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData
from scipy import sparse

from sckitflow.data._group_encoders import LabelEncoderConfig, OneHotEncoderConfig
from sckitflow.data._loader import Loader
from sckitflow.data._manager import DataManager, DataManagerConfig
from sckitflow.data.schemas import CouplingDataSchema

LINES = ["s0", "s1"]
DRUGS = ["control", "d0", "d1"]
CELLS_PER_COMBO = 12
DRUG_DIM, LINE_DIM, COND_DIM, TGT_DIM, REP_DIM = 4, 3, 6, 2, 9


@pytest.fixture
def loader_adata() -> AnnData:
    """Small deterministic AnnData: groups=source_split, condition=drugA, + continuous covs and an obsm rep."""
    rng = np.random.default_rng(0)
    rows = [(line, drug) for line in LINES for drug in DRUGS for _ in range(CELLS_PER_COMBO)]
    obs = pd.DataFrame(rows, columns=["source_split", "drugA"]).astype("category")
    n = len(obs)
    ad = AnnData(X=rng.standard_normal((n, 7)).astype(np.float32), obs=obs)
    ad.uns["drug"] = {d: rng.standard_normal((1, DRUG_DIM)).astype(np.float32) for d in DRUGS}
    ad.uns["source_split"] = {line: rng.standard_normal((1, LINE_DIM)).astype(np.float32) for line in LINES}
    ad.obsm["Xcond"] = rng.standard_normal((n, COND_DIM)).astype(np.float32)  # continuous condition cov
    ad.obsm["ytgt"] = rng.standard_normal((n, TGT_DIM)).astype(np.float32)  # continuous target cov
    ad.obsm["Xrep"] = rng.standard_normal((n, REP_DIM)).astype(np.float32)  # alternate state rep
    ad.obsm["Xcond"] = rng.standard_normal((n, COND_DIM)).astype(np.float32)  # continuous condition cov
    ad.obsm["ytgt"] = rng.standard_normal((n, TGT_DIM)).astype(np.float32)  # continuous target cov
    ad.obsm["Xrep"] = rng.standard_normal((n, REP_DIM)).astype(np.float32)  # alternate state rep
    # Coupling reps: shared linear (5) + per-side quadratics (3 / 4), all different widths so a
    # bug that swaps sides or reads one width twice fails loudly.
    ad.obsm["X_lin"] = rng.standard_normal((n, 5)).astype(np.float32)
    ad.obsm["X_src_quad"] = rng.standard_normal((n, 3)).astype(np.float32)
    ad.obsm["X_tgt_quad"] = rng.standard_normal((n, 4)).astype(np.float32)
    return ad


def _dm(**overrides) -> DataManager:
    base = {
        "categorical_conditions": {"drug": ("drugA",)},
        "categorical_reps": {"drug": "drug"},
        "continuous_conditions": ["Xcond"],
        "groups": ("source_split",),
        "groups_reps": {"source_split": "source_split"},
        "control_values_dict": {"drug": "control"},
        "target_continuous_covs": ["ytgt"],
    }
    base.update(overrides)
    return DataManager(DataManagerConfig(**base))


def _loader(ad, dm, **overrides) -> Loader:
    kwargs = {
        "primary_weights": {("s0", "d0"): 1.0},
        "control_weights": {("s0", "control"): 1.0, ("s1", "control"): 1.0},
        "batch_size": 8,
    }
    kwargs.update(overrides)
    return Loader(ad, **dm._loader_schema, **kwargs)


class TestLoaderStepDataMapping:
    def test_weights_select_group_and_label_gid_encoding(self, loader_adata):
        """primary_weights pick the group; its categorical encoding is looked up by gid, one row for the batch."""
        ad = loader_adata
        loader = _loader(ad, _dm())
        batches = list(loader)
        assert batches, "loader must yield at least one batch"
        for sd in batches:
            grp = sd["target_group_data"]["source_split"]
            drug = sd["target_group_data"]["drug"]  # categorical conditions describe the group too
            # (1, c, d): one row, broadcast to every observation of the batch. `c` is the columns mapped to the
            # realm -- 1 here -- and is the axis the `SetEncoder` pools over.
            assert grp.shape == (1, 1, LINE_DIM)
            assert drug.shape == (1, 1, DRUG_DIM)
            # and that single encoding is exactly the *selected* group's uns rep (weights => (s0, d0))
            # `[0]` picks a row, `.ravel()` flattens the single-column set axis to compare the values
            np.testing.assert_allclose(grp[0].numpy().ravel(), ad.uns["source_split"]["s0"].ravel(), rtol=1e-6)
            np.testing.assert_allclose(drug[0].numpy().ravel(), ad.uns["drug"]["d0"].ravel(), rtol=1e-6)

    def test_continuous_covariates_ride_as_per_cell_reps(self, loader_adata):
        """Continuous condition/target covariates stream per-cell straight from obsm (not tiled)."""
        ad = loader_adata
        sd = next(iter(_loader(ad, _dm())))
        b = sd["target_state"].shape[0]
        xc = sd["target_condition_data"]["Xcond"]
        yt = sd["target_response_data"]["ytgt"]
        assert xc.shape == (b, COND_DIM)
        assert yt.shape == (b, TGT_DIM)
        # provenance: every streamed row is one of the selected group's obsm rows (per-cell, not a tile)
        mask = ((ad.obs["source_split"].astype(str) == "s0") & (ad.obs["drugA"].astype(str) == "d0")).to_numpy()
        src = ad.obsm["Xcond"][mask]
        for row in xc.numpy():
            assert np.any(np.all(np.isclose(src, row), axis=1)), "Xcond row not from the selected group"

    def test_sample_rep_reads_state_from_obsm(self, loader_adata):
        """With target_sample_rep set, target_state comes from that obsm key instead of .X."""
        ad = loader_adata
        sd = next(iter(_loader(ad, _dm(target_sample_rep="Xrep"))))
        assert sd["target_state"].shape[1] == REP_DIM

    def test_control_link_populates_source_state(self, loader_adata):
        """A control link (same-adata weights) fills source_state with the matched control population."""
        ad = loader_adata
        sd = next(iter(_loader(ad, _dm())))
        assert sd["source_state"] is not None
        assert sd["source_state"].shape[1] == sd["target_state"].shape[1]

    def test_every_step_data_key_is_present(self, loader_adata):
        """The emitted dict carries every StepData key, so consumers can index without guarding."""
        from sckitflow.data._loader import _STEP_DATA_KEYS

        sd = next(iter(_loader(loader_adata, _dm())))
        assert set(_STEP_DATA_KEYS) <= set(sd)


class TestLoaderIterationContract:
    def test_iter_is_finite_and_reiterable(self, loader_adata):
        """__iter__ yields exactly one epoch (len) and can be iterated again -- scfit's is infinite."""
        loader = _loader(loader_adata, _dm())
        n = len(loader)
        assert n >= 1
        first = list(loader)
        second = list(loader)
        assert len(first) == n
        assert len(second) == n

    def test_n_iters_sets_pass_length(self, loader_adata):
        """n_iters bounds one pass to that many batches (forwarded to scfit); the trainer iterates it."""
        loader = _loader(loader_adata, _dm(), n_iters=3)
        assert len(loader) == 3
        assert sum(1 for _ in loader) == 3

    def test_schema_without_covariates_streams_one_implicit_group(self, loader_adata):
        """An unconditional schema still streams: every cell is in the one group ``()``."""
        ad = loader_adata
        before = list(ad.obs.columns)
        loader = Loader(ad, **DataManager()._loader_schema, batch_size=8)
        sd = next(iter(loader))

        assert sd["target_state"].shape == (8, ad.n_vars)
        assert sd["source_state"] is None  # nothing to pair against
        assert sd["target_condition_data"] is None and sd["target_group_data"] is None
        assert loader.group_cols == () and loader._leaves == [()]  # one group covering every cell
        assert list(ad.obs.columns) == before  # the caller's obs gains no column


class TestSourceAlignment:
    """The source is row-aligned to the batch, so every per-cell field of a StepData agrees in length."""

    def test_source_is_sliced_when_longer_and_tiled_when_shorter(self, loader_adata):
        loader = _loader(loader_adata, _dm())
        src = torch.arange(3.0).unsqueeze(1)  # three control observations

        assert loader._align_source(src, 3) is src  # already aligned: no gather at all
        assert loader._align_source(src, 2).squeeze(1).tolist() == [0.0, 1.0]
        assert loader._align_source(src, 5).squeeze(1).tolist() == [0.0, 1.0, 2.0, 0.0, 1.0]

    def test_no_source_is_left_alone(self, loader_adata):
        """Unpaired (generate-from-noise) batches have no source to align."""
        assert _loader(loader_adata, _dm())._align_source(None, 4) is None

    def test_zero_matched_controls_raise(self, loader_adata):
        """A group with no matched control has nothing to flow from -- that is a data problem, not a 0-row batch."""
        loader = _loader(loader_adata, _dm())
        with pytest.raises(ValueError, match="zero control observations"):
            loader._align_source(torch.empty(0, 7), 4)


class TestZeroCopyConversion:
    """Batches stream as annbatch's native arrays (`to=None`) and become torch without a copy."""

    def test_numpy_batches_share_memory_with_the_tensor(self):
        from sckitflow.data._loader import _as_tensor

        array = np.zeros((2, 3), dtype=np.float32)
        tensor = _as_tensor(array)
        tensor[0, 0] = 7.0
        assert array[0, 0] == 7.0, "conversion copied instead of viewing the buffer"

    def test_an_existing_tensor_passes_straight_through(self):
        from sckitflow.data._loader import _as_tensor

        tensor = torch.zeros(2, 3)
        assert _as_tensor(tensor) is tensor

    def test_a_non_dlpack_object_is_refused_by_name(self):
        """A non-array (sparse, pandas, None) must name itself rather than raise from inside torch."""
        from sckitflow.data._loader import _as_tensor

        with pytest.raises(TypeError, match=r"streamed csr_matrix.*DLPack"):
            _as_tensor(sparse.csr_matrix((2, 3), dtype=np.float32))

    def test_a_dlpack_array_converts(self):
        """Anything exposing ``__dlpack__`` passes the guard -- the cupy path, exercised via numpy."""
        from sckitflow.data._loader import _as_tensor

        class OnlyDLPack:
            """A numpy array behind a wrapper exposing *only* the DLPack protocol."""

            def __init__(self, array):
                self._array = array

            def __dlpack__(self, *args, **kwargs):
                return self._array.__dlpack__(*args, **kwargs)

            def __dlpack_device__(self):
                return self._array.__dlpack_device__()

        tensor = _as_tensor(OnlyDLPack(np.ones((2, 3), dtype=np.float32)))
        assert tensor.shape == (2, 3)


class TestLeafEncoding:
    """A group encoding comes off a single representative cell, so its leading axis must be that one cell."""

    def test_one_dimensional_passes_through(self):
        from sckitflow.data._loader import _leaf_encoding

        assert _leaf_encoding(np.arange(3)).shape == (3,)

    def test_only_the_leading_cell_axis_is_dropped(self):
        """`CategoricalData.extract_reps` stacks a realm's columns, so encodings are `(1, n_cols, dim)`."""
        from sckitflow.data._loader import _leaf_encoding

        assert _leaf_encoding(np.zeros((1, 1, 3))).shape == (1, 3)
        assert _leaf_encoding(np.zeros((1, 2, 3))).shape == (2, 3)

    def test_a_rank_that_extract_reps_cannot_emit_is_refused(self):
        """Only rank 3 (or a flat rank-1 rep) is reachable: `np.stack` always adds the set axis."""
        from sckitflow.data._loader import _leaf_encoding

        for bad in [(1, 3), (1, 2, 3, 4)]:
            with pytest.raises(ValueError, match=r"rank 1 or 3"):
                _leaf_encoding(np.zeros(bad))

    def test_group_row_refuses_an_undropped_cell_axis(self):
        """Rank 3 into `_group_row` means `_leaf_encoding` was skipped; the encoder would pool the wrong axis."""
        from sckitflow.data._loader import _group_row

        with pytest.raises(ValueError, match=r"rank 1 or 2"):
            _group_row(torch.zeros(1, 2, 3))

    def test_multiple_rows_raise_instead_of_silently_truncating(self):
        from sckitflow.data._loader import _leaf_encoding

        with pytest.raises(ValueError, match=r"one representative row.*got 2"):
            _leaf_encoding(np.zeros((2, 3)))

    @pytest.mark.parametrize("pooling", ["mean", "sum"])
    def test_a_group_row_feeds_the_set_encoder_for_any_realm_width(self, pooling):
        """The loader -> `SetEncoder` shape contract, which nothing else covers end to end.

        A group's encoding is `(1, c, d)`; pooling reduces `c` (`dim=-2`), so the set axis must survive and the
        batch axis stay 1. Losing `c` would make the pooling reduce the wrong axis, silently.
        """
        from sckitflow.core.nn.blocks._mlp import MLPConfig, MLPEncoderConfig
        from sckitflow.core.nn.blocks._pooling import MeanPoolingConfig, SumPoolingConfig
        from sckitflow.core.nn.blocks._set_encoder import SetEncoderConfig
        from sckitflow.data._loader import _group_row, _leaf_encoding

        dim, out_dim = 3, 8
        encoder = SetEncoderConfig(
            output_dim=out_dim,
            categorical_input_layers={"drug": MLPEncoderConfig(mlp=MLPConfig(), output_dim=out_dim)},
            pooling={"mean": MeanPoolingConfig, "sum": SumPoolingConfig}[pooling](),
            pooling_proj_dim=out_dim,
        ).build(categorical_dims={"drug": dim}, continuous_dims={})
        for n_cols in (1, 2, 3):
            # the shape `CategoricalData.extract_reps` emits for a realm of `n_cols` columns
            rep = np.zeros((1, n_cols, dim), dtype=np.float32)
            row = _group_row(torch.as_tensor(_leaf_encoding(rep)))
            assert tuple(row.shape) == (1, n_cols, dim), f"set axis lost for c={n_cols}"
            assert tuple(encoder({"drug": row}).shape) == (1, out_dim), f"batch axis lost for c={n_cols}"

    def test_a_real_single_column_encoding_is_one_group_row(self):
        """The shape actually produced by the encoders survives `_leaf_encoding` -> `_group_row` intact."""
        import pandas as pd

        from sckitflow.data._loader import _group_row, _leaf_encoding
        from sckitflow.data._utils import get_one_hot_encoder
        from sckitflow.data.containers._categorical import CategoricalData

        enc = get_one_hot_encoder(np.array(["d0", "d1", "control"]).reshape(-1, 1))
        cd = CategoricalData.from_pandas(
            pd.DataFrame({"drug": ["d0"]}),
            categorical_encoders={"drug": enc},
            categorical_reps_map={"drug": "drug"},
        )
        (rep,) = cd.extract_reps().mapping.values()
        assert np.asarray(rep).shape == (1, 1, 3), "encoder contract changed"
        assert tuple(_group_row(torch.as_tensor(_leaf_encoding(rep))).shape) == (1, 1, 3)


class TestLoaderDtypeAndDevice:
    """The last loading stage settles dtype/device, so batches reach the method ready to consume."""

    def test_float64_source_is_cast_to_the_method_dtype(self, loader_adata):
        """A float64 source would otherwise hit float32 modules ('mat1 and mat2 must have the same dtype')."""
        ad = loader_adata
        ad.X = ad.X.astype(np.float64)
        ad.obsm["Xcond"] = ad.obsm["Xcond"].astype(np.float64)
        sd = next(iter(_loader(ad, _dm(), dtype=torch.float32, device="cpu")))
        assert sd["target_state"].dtype is torch.float32
        assert sd["source_state"].dtype is torch.float32
        assert sd["target_condition_data"]["Xcond"].dtype is torch.float32
        # the per-group encodings go through the same conform step
        assert sd["target_group_data"]["source_split"].dtype is torch.float32

    def test_no_dtype_leaves_the_streamed_tensors_untouched(self, loader_adata):
        """Opting out is the default: nothing is cast or copied when neither dtype nor device is set."""
        ad = loader_adata
        ad.X = ad.X.astype(np.float64)
        sd = next(iter(_loader(ad, _dm())))
        assert sd["target_state"].dtype is torch.float64

    def test_batches_on_the_wrong_device_raise(self, loader_adata, monkeypatch):
        """A per-batch device copy is a real problem, so it fails loudly instead of being paid for."""
        loader = _loader(loader_adata, _dm(), device="cpu")
        # pretend the method runs elsewhere: batches stream on CPU, so every one would be copied
        monkeypatch.setattr(loader, "_device_type", "cuda")
        with pytest.raises(RuntimeError, match="copied across devices"):
            next(iter(loader))

    def test_host_encodings_are_moved_not_asserted(self, loader_adata, monkeypatch):
        """`uns` encodings are host arrays by construction, so moving them is the only option."""
        loader = _loader(loader_adata, _dm(), device="cpu")
        monkeypatch.setattr(loader, "_device_type", "cuda")  # `_device` stays "cpu" so the move is a no-op
        with pytest.raises(RuntimeError):
            loader._conform(torch.zeros(2, 2))
        assert loader._conform(torch.zeros(2, 2), streamed=False).device.type == "cpu"


class TestCouplingLocs:
    """`_get_coupling_locs` derives rep locs from the schema, one per configured term."""

    def test_none_schema_yields_empty_locs(self):
        from sckitflow.data._loader import _get_coupling_locs

        assert _get_coupling_locs(None) == ({}, {})

    def test_linear_only_default_rep_lin_aliases_x(self):
        """`rep_lin=None` means `.X`, so both sides point at `"X"`."""
        from sckitflow.data._loader import _get_coupling_locs

        src, tgt = _get_coupling_locs(CouplingDataSchema())
        assert src == {"X": "source_coupling_lin"}
        assert tgt == {"X": "target_coupling_lin"}

    def test_linear_only_with_obsm_rep(self):
        from sckitflow.data._loader import _get_coupling_locs

        src, tgt = _get_coupling_locs(CouplingDataSchema(rep_lin="X_lin"))
        assert src == {"obsm/X_lin": "source_coupling_lin"}
        assert tgt == {"obsm/X_lin": "target_coupling_lin"}

    def test_fused_carries_shared_lin_and_per_side_quads(self):
        from sckitflow.data._loader import _get_coupling_locs

        schema = CouplingDataSchema(
            rep_lin="X_lin",
            source_rep_quad="X_src_quad",
            target_rep_quad="X_tgt_quad",
        )
        src, tgt = _get_coupling_locs(schema)
        assert src == {"obsm/X_lin": "source_coupling_lin", "obsm/X_src_quad": "source_coupling_quad"}
        assert tgt == {"obsm/X_lin": "target_coupling_lin", "obsm/X_tgt_quad": "target_coupling_quad"}

    def test_pure_gw_has_no_linear_locs(self):
        from sckitflow.data._loader import _get_coupling_locs

        schema = CouplingDataSchema(
            use_linear=False,
            source_rep_quad="X_src_quad",
            target_rep_quad="X_tgt_quad",
        )
        src, tgt = _get_coupling_locs(schema)
        assert src == {"obsm/X_src_quad": "source_coupling_quad"}
        assert tgt == {"obsm/X_tgt_quad": "target_coupling_quad"}


class TestMergeReps:
    """`_merge_reps` dedupes rep locs -- the shared linear term aliases the state by default."""

    def test_dedupe_preserves_first_occurrence_order(self):
        from sckitflow.data._loader import _merge_reps

        assert _merge_reps(("a", "b"), ("b", "c"), ("a",)) == ("a", "b", "c")

    def test_empty_groups(self):
        from sckitflow.data._loader import _merge_reps

        assert _merge_reps() == ()
        assert _merge_reps({}, ()) == ()

    def test_dict_keys_are_iterated(self):
        from sckitflow.data._loader import _merge_reps

        assert _merge_reps({"a": 1, "b": 2}, ("c",)) == ("a", "b", "c")


class TestLoaderCouplingStreaming:
    """Loader streams coupling reps alongside the state, one StepData key per configured term."""

    def _coupling_loader(self, ad, schema, **overrides):
        """Same-adata control weights: no separate pool, but the source side still carries coupling."""
        return _loader(
            ad,
            _dm(),
            coupling_schema=schema,
            primary_weights={("s0", "d0"): 1.0},
            control_weights={("s0", "control"): 1.0},
            **overrides,
        )

    def test_no_schema_leaves_every_coupling_key_none(self, loader_adata):
        """The default: no schema → all four coupling keys are None."""
        sd = next(iter(_loader(loader_adata, _dm())))
        assert sd["target_coupling_lin"] is None
        assert sd["target_coupling_quad"] is None
        assert sd["source_coupling_lin"] is None
        assert sd["source_coupling_quad"] is None

    def test_linear_only_default_aliases_state(self, loader_adata):
        """`rep_lin=None`: the linear coupling term *is* the state, on both sides."""
        sd = next(iter(self._coupling_loader(loader_adata, CouplingDataSchema())))
        assert sd["target_coupling_lin"] is not None
        assert sd["source_coupling_lin"] is not None
        assert sd["target_coupling_quad"] is None
        assert sd["source_coupling_quad"] is None
        torch.testing.assert_close(sd["target_coupling_lin"], sd["target_state"])
        torch.testing.assert_close(sd["source_coupling_lin"], sd["source_state"])

    def test_linear_only_with_obsm_rep_has_lin_width(self, loader_adata):
        sd = next(iter(self._coupling_loader(loader_adata, CouplingDataSchema(rep_lin="X_lin"))))
        b = sd["target_state"].shape[0]
        assert sd["target_coupling_lin"].shape == (b, 5)
        assert sd["source_coupling_lin"].shape == (b, 5)
        assert sd["target_coupling_quad"] is None
        assert sd["source_coupling_quad"] is None

    def test_fused_streams_shared_lin_and_per_side_quads(self, loader_adata):
        """Each side reads its own quadratic -- source 3-wide, target 4-wide -- over a shared 5-wide lin."""
        schema = CouplingDataSchema(
            rep_lin="X_lin",
            source_rep_quad="X_src_quad",
            target_rep_quad="X_tgt_quad",
        )
        sd = next(iter(self._coupling_loader(loader_adata, schema)))
        b = sd["target_state"].shape[0]
        assert sd["target_coupling_lin"].shape == (b, 5)
        assert sd["source_coupling_lin"].shape == (b, 5)
        assert sd["target_coupling_quad"].shape == (b, 4)
        assert sd["source_coupling_quad"].shape == (b, 3)

    def test_pure_gw_omits_linear_on_both_sides(self, loader_adata):
        schema = CouplingDataSchema(
            use_linear=False,
            source_rep_quad="X_src_quad",
            target_rep_quad="X_tgt_quad",
        )
        sd = next(iter(self._coupling_loader(loader_adata, schema)))
        assert sd["target_coupling_lin"] is None
        assert sd["source_coupling_lin"] is None
        assert sd["target_coupling_quad"].shape[1] == 4
        assert sd["source_coupling_quad"].shape[1] == 3

    def test_source_and_target_quads_are_independent_reps(self, loader_adata):
        """The `_target_coupling_dict` iterating the source dicts bug would `KeyError` here: with a
        fused schema the source and target quad locs are disjoint, and the widths differ on purpose."""
        schema = CouplingDataSchema(
            rep_lin="X_lin",
            source_rep_quad="X_src_quad",
            target_rep_quad="X_tgt_quad",
        )
        sd = next(iter(self._coupling_loader(loader_adata, schema)))
        assert sd["source_coupling_quad"].shape[1] == 3
        assert sd["target_coupling_quad"].shape[1] == 4

    def test_unpaired_with_a_coupling_schema_still_streams_target(self, loader_adata):
        """No control link ⇒ source coupling is None; target coupling still streams (from the primary)."""
        schema = CouplingDataSchema(rep_lin="X_lin", target_rep_quad="X_tgt_quad")
        sd = next(
            iter(
                _loader(
                    loader_adata,
                    _dm(),
                    coupling_schema=schema,
                    primary_weights={("s0", "d0"): 1.0},
                    control_weights=None,
                )
            )
        )
        assert sd["source_coupling_lin"] is None
        assert sd["source_coupling_quad"] is None
        assert sd["target_coupling_lin"].shape[1] == 5
        assert sd["target_coupling_quad"].shape[1] == 4


class TestEvalLoaderNeverStreamsCoupling:
    """EvalLoader's signature *is* the contract: no `coupling_schema`, so no coupling in the batch."""

    def test_does_not_accept_coupling_schema(self):
        """Removing the parameter is how the training-only invariant is enforced at the API boundary."""
        import inspect

        from sckitflow.data._loader import EvalLoader

        params = inspect.signature(EvalLoader.__init__).parameters
        assert "coupling_schema" not in params, (
            "EvalLoader must not accept coupling_schema: coupling is a training-time concern. "
            "If eval-time coupling is ever needed, add `stream_coupling: bool = False` explicitly "
            "rather than a bare schema (see the class docstring)."
        )

    def test_eval_batches_carry_no_coupling_keys(self, loader_adata):
        """All four coupling keys are None, matching the class docstring's promise."""
        from sckitflow.data._loader import EvalLoader

        el = EvalLoader(
            loader_adata,
            **_dm()._loader_schema,  # note: shared schema, not `_training_loader_schema`
            primary_weights={("s0", "d0"): 1.0},
            control_weights={("s0", "control"): 1.0},
        )
        for sd, _leaf in el:
            assert sd["target_coupling_lin"] is None
            assert sd["target_coupling_quad"] is None
            assert sd["source_coupling_lin"] is None
            assert sd["source_coupling_quad"] is None


class TestGroupEncoding:
    """Encoded groups (no `uns` rep) stream with the config's vocabulary, whatever cell a group is encoded from."""

    @pytest.mark.parametrize(
        "encoder", [LabelEncoderConfig(classes=("s0", "s1")), OneHotEncoderConfig(categories=("s0", "s1"))]
    )
    def test_each_group_gets_its_own_code(self, loader_adata, encoder):
        dm = _dm(groups_reps=None, groups_encoding={"source_split": encoder})
        codes = {
            split: next(iter(_loader(loader_adata, dm, primary_weights={(split, "d0"): 1.0})))["target_group_data"][
                "source_split"
            ][0]
            for split in ("s0", "s1")
        }
        assert not torch.equal(codes["s0"], codes["s1"])
        dims = dm.get_data_dimensionalities(loader_adata)
        assert codes["s0"].shape[-1] == dims.groups_reps_dims["source_split"]
