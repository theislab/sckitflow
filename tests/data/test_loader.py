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
        "to": "torch",
        "batch_size": 8,
    }
    kwargs.update(overrides)
    return Loader(ad, dm=dm, **kwargs)


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
        mask = (
            (ad.obs["source_split"].astype(str) == "s0") & (ad.obs["drugA"].astype(str) == "d0")
        ).to_numpy()
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

    def test_sample_returns_one_batch_and_cycles_past_epoch(self, loader_adata):
        """sample() yields a 1-tuple StepData and keeps going across epoch boundaries."""
        loader = _loader(loader_adata, _dm())
        n = len(loader)
        got = [loader.sample() for _ in range(2 * n + 3)]  # more than one epoch
        assert all(len(batch) == 1 for batch in got)
        assert all(batch[0]["target_state"] is not None for batch in got)

    def test_needs_a_categorical_group_or_condition_column(self, loader_adata):
        """A schema with no group/condition column cannot be grouped on -> explicit error."""
        with pytest.raises(ValueError, match="at least one categorical"):
            Loader(loader_adata, dm=DataManager(), to="torch")
