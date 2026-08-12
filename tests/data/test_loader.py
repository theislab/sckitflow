"""Unit tests for :class:`sckitflow.data._loader.Loader`.

Covers the scfit-batch -> ``StepData`` bridge: weight-based group selection, the label-gid ->
cached categorical encoding (tiled to the batch), continuous covariates riding as per-cell reps,
and the finite/cycling iteration contract the trainer relies on.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData

from sckitflow.data._loader import Loader
from sckitflow.data._manager import DataManager

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
    return ad


def _dm(**overrides) -> DataManager:
    base = {
        "conditions": {"drug": ("drugA",)},
        "conditions_reps": {"drug": "drug"},
        "conditions_covariates": ["Xcond"],
        "groups": ("source_split",),
        "groups_reps": {"source_split": "source_split"},
        "control_values_dict": {"drug": "control"},
        "target_continuous_covs": ["ytgt"],
    }
    base.update(overrides)
    return DataManager(**base)


def _loader(ad, dm, **overrides) -> Loader:
    kwargs = {
        "primary_weights": {("s0", "d0"): 1.0},
        "control_weights": {("s0", "control"): 1.0, ("s1", "control"): 1.0},
        "batch_size": 8,
    }
    kwargs.update(overrides)
    return Loader(ad, **dm._loader_schema, **kwargs)


class TestLoaderStepDataMapping:
    def test_weights_select_group_and_label_gid_tiles_encoding(self, loader_adata):
        """primary_weights pick the group; its categorical encoding is looked up by gid and tiled."""
        ad = loader_adata
        loader = _loader(ad, _dm())
        batches = list(loader)
        assert batches, "loader must yield at least one batch"
        for sd in batches:
            b = sd["target_state"].shape[0]
            grp = sd["target_group_data"]["source_split"]
            drug = sd["target_condition_data"]["drug"]
            assert grp.shape == (b, LINE_DIM)
            assert drug.shape == (b, DRUG_DIM)
            # tiled: label-gid -> one cached encoding -> broadcast to every row
            assert torch.allclose(grp, grp[0].expand_as(grp))
            assert torch.allclose(drug, drug[0].expand_as(drug))
            # and that single encoding is exactly the *selected* group's uns rep (weights => (s0, d0))
            np.testing.assert_allclose(grp[0].numpy(), ad.uns["source_split"]["s0"].ravel(), rtol=1e-6)
            np.testing.assert_allclose(drug[0].numpy(), ad.uns["drug"]["d0"].ravel(), rtol=1e-6)

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
        """With sample_rep set, target_state comes from that obsm key instead of .X."""
        ad = loader_adata
        sd = next(iter(_loader(ad, _dm(sample_rep="Xrep"))))
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
        """An unconditional schema still streams: every cell lands in one synthesized group."""
        from sckitflow.data._loader import _ALL_CELLS

        ad = loader_adata
        before = list(ad.obs.columns)
        loader = Loader(ad, **DataManager()._loader_schema, batch_size=8)
        sd = next(iter(loader))

        assert sd["target_state"].shape == (8, ad.n_vars)
        assert sd["source_state"] is None  # nothing to pair against
        assert sd["target_condition_data"] is None and sd["target_group_data"] is None
        assert loader.group_cols == (_ALL_CELLS,)  # one group covering every cell
        assert list(ad.obs.columns) == before  # ...synthesized on a shallow copy, not the caller's obs


class TestSourceAlignment:
    """The source is row-aligned to the batch, so every per-cell field of a StepData agrees in length."""

    def test_source_is_sliced_when_longer_and_tiled_when_shorter(self, loader_adata):
        loader = _loader(loader_adata, _dm())
        src = torch.arange(3.0).unsqueeze(1)  # three control cells

        assert loader._align_source(src, 3) is src  # already aligned: no gather at all
        assert loader._align_source(src, 2).squeeze(1).tolist() == [0.0, 1.0]
        assert loader._align_source(src, 5).squeeze(1).tolist() == [0.0, 1.0, 2.0, 0.0, 1.0]

    def test_no_source_is_left_alone(self, loader_adata):
        """Unpaired (generate-from-noise) batches have no source to align."""
        assert _loader(loader_adata, _dm())._align_source(None, 4) is None

    def test_zero_matched_controls_raise(self, loader_adata):
        """A group with no matched control has nothing to flow from -- that is a data problem, not a 0-row batch."""
        loader = _loader(loader_adata, _dm())
        with pytest.raises(ValueError, match="zero control cells"):
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
        # the tiled per-group encodings go through the same conform step
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
