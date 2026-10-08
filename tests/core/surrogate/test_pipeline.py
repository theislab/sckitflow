# tests/integration/test_surrogate_potential_integration.py
"""
Integration tests for ``SurrogatePotential`` on the comprehensive dummy
AnnData fixture.

The DataManager, FlowDataModule, EvalLoader and StepData pipeline are all
real. Only the inferer is faked, so the tests are fast, deterministic and
independent of any trained weights. The potential's contract is with the
shapes, not with the model.
"""

from __future__ import annotations

import pytest
import torch

from sckitflow.core._types import PredictionData
from sckitflow.core.surrogate import SurrogatePotential
from sckitflow.data import DataManagerConfig, FlowDataModule

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONT_COV_1 = "drugA_time"
CONT_COV_2 = "drugA_dose"
CONT_COV_WIDE = "paired_condition"
CONT_COV_WIDE_DIM = 100


# ---------------------------------------------------------------------------
# Fake inferer
# ---------------------------------------------------------------------------
class FakeInferer:
    """A minimal stand-in for a trained inference method.

    The response is a differentiable function of the conditions — the mean of
    the first continuous covariate, broadcast to ``response_dim``. Its batch
    layout mirrors whatever it receives, so the same fake covers the
    unaligned ``(N, D)``, the aligned ``(B, N, D)``, and the multi-sample
    ``(M, ...)`` cases without any branching.
    """

    def __init__(self, response_dim: int, *, n_samples: int = 1):
        self._module = torch.nn.Linear(1, 1)
        self.response_dim = response_dim
        self.n_samples = n_samples
        self.calls: list = []

    @property
    def module(self) -> torch.nn.Module:
        return self._module

    def predict(self, step_data, *, generator):
        self.calls.append(step_data)
        conds = step_data["target_condition_data"]
        candidates = [v for v in conds.values() if isinstance(v, torch.Tensor) and v.ndim >= 2]
        if not candidates:
            raise RuntimeError("FakeInferer: no per-sample covariate")

        # The aligned layout is (B, N, D); its condition axis lives at
        # position -2 and survives ``get_response``'s collapse. Prefer a 3D
        # tensor so the fake never accidentally uses the batch dim as N.
        three_d = [v for v in candidates if v.ndim == 3]
        if three_d:
            ref = three_d[0]
        else:
            # Unaligned path: the batch and condition axes coincide, so the
            # reference is the covariate with the largest leading dim.
            ref = max(candidates, key=lambda v: v.shape[0])

        sig = ref.mean(dim=-1, keepdim=True)
        sig = sig.expand(*sig.shape[:-1], self.response_dim)

        # The seed must affect the response, otherwise the determinism
        # tests cannot distinguish a different-seed call from a same-seed one.
        if self.n_samples > 1:
            noise = torch.randn(self.n_samples, *sig.shape, generator=generator)
            sig = sig.unsqueeze(0) + 0.1 * noise
        else:
            sig = sig + 0.1 * torch.randn_like(sig, generator=generator)

        return PredictionData(X=None, traj=None, raw_samples=sig)


# ---------------------------------------------------------------------------
# Concrete potential
# ---------------------------------------------------------------------------


class MeanMSESurrogate(SurrogatePotential):
    """Collapses every leading axis of ``raw_samples`` down to ``(N, R)``,
    then returns a per-condition squared-error potential of shape ``(N,)``."""

    def get_response(self, pred_data: PredictionData) -> torch.Tensor:
        x = pred_data.raw_samples
        while x.ndim > 2:
            x = x.mean(dim=0)
        return x

    def compute_raw_potential(self, y, ystar):
        return ((y - ystar) ** 2).sum(dim=-1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_datamodule(
    adata,
    *,
    sample_rep=None,
    conditions=None,
    conditions_reps=None,
    conditions_covariates=None,
    groups=None,
    groups_encoding=None,
    control_values_dict=None,
    matched_pairs=None,
    batch_size=64,
) -> FlowDataModule:
    config = DataManagerConfig(
        sample_rep=sample_rep,
        conditions=conditions,
        conditions_reps=conditions_reps,
        conditions_covariates=conditions_covariates,
        groups=groups,
        groups_encoding=groups_encoding,
        control_values_dict=control_values_dict,
        matched_pairs=matched_pairs,
    ).build()
    return FlowDataModule.from_adata(adata, config, n_train_steps=1, batch_size=batch_size)


def build_potential(
    adata,
    cond_dict,
    ystar,
    *,
    n_samples: int = 1,
    require_target_state: bool = True,
    potential_seed: int = 0,
    **dmod_kwargs,
):
    dmod = build_datamodule(adata, **dmod_kwargs)
    dmod.set_predict_data(adata, require_target_state=require_target_state)
    inferer = FakeInferer(ystar.numel(), n_samples=n_samples)
    potential = MeanMSESurrogate(inferer, ystar, dmod, seed=potential_seed)
    return potential, dmod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ystar(n_genes):
    return torch.zeros(n_genes)


@pytest.fixture
def cond_dict_single():
    return {CONT_COV_1: torch.zeros((4, 1), requires_grad=True)}


@pytest.fixture
def cond_dict_pair():
    return {
        CONT_COV_1: torch.zeros((4, 1), requires_grad=True),
        CONT_COV_2: torch.zeros((4, 1), requires_grad=True),
    }


@pytest.fixture
def cond_dict_wide():
    return {CONT_COV_WIDE: torch.zeros((4, CONT_COV_WIDE_DIM), requires_grad=True)}


# ---------------------------------------------------------------------------
# Minimal configuration
# ---------------------------------------------------------------------------


class TestMinimalConfig:
    def test_forward_returns_per_condition_vector(self, adata, cond_dict_single, ystar):
        potential, _ = build_potential(
            adata,
            cond_dict_single,
            ystar,
            conditions_covariates=(CONT_COV_1,),
        )
        out = potential(cond_dict_single)
        assert out.shape == (4,)
        assert out.requires_grad

    def test_gradient_reaches_conditions(self, adata, ystar):
        cond_dict = {CONT_COV_1: torch.randn(4, 1, requires_grad=True)}
        potential, _ = build_potential(
            adata,
            cond_dict,
            ystar,
            conditions_covariates=(CONT_COV_1,),
        )
        out = potential(cond_dict)
        out.sum().backward()
        assert cond_dict[CONT_COV_1].grad is not None
        assert cond_dict[CONT_COV_1].grad.shape == cond_dict[CONT_COV_1].shape

    def test_two_continuous_covariates(self, adata, cond_dict_pair, ystar):
        potential, _ = build_potential(
            adata,
            cond_dict_pair,
            ystar,
            conditions_covariates=(CONT_COV_1, CONT_COV_2),
        )
        out = potential(cond_dict_pair)
        assert out.shape == (4,)

    def test_wide_continuous_covariate(self, adata, cond_dict_wide, ystar):
        potential, _ = build_potential(
            adata,
            cond_dict_wide,
            ystar,
            conditions_covariates=(CONT_COV_WIDE,),
        )
        out = potential(cond_dict_wide)
        assert out.shape == (4,)


# ---------------------------------------------------------------------------
# generate_from_noise — True vs False
# ---------------------------------------------------------------------------


class TestGenerateFromNoise:
    @pytest.mark.parametrize("n_samples", [1, 2, 4, 16])
    def test_sample_count_does_not_change_output_shape(self, adata, cond_dict_single, ystar, n_samples):
        potential, _ = build_potential(
            adata,
            cond_dict_single,
            ystar,
            n_samples=n_samples,
            conditions_covariates=(CONT_COV_1,),
        )
        out = potential(cond_dict_single)
        assert out.shape == (4,)

    @pytest.mark.parametrize("n_samples", [1, 8])
    def test_gradient_reaches_conditions(self, adata, ystar, n_samples):
        cond_dict = {CONT_COV_1: torch.randn(4, 1, requires_grad=True)}
        potential, _ = build_potential(
            adata,
            cond_dict,
            ystar,
            n_samples=n_samples,
            conditions_covariates=(CONT_COV_1,),
        )
        out = potential(cond_dict)
        out.sum().backward()
        assert cond_dict[CONT_COV_1].grad is not None

    def test_single_sample_and_many_samples_differ(self, adata, ystar):
        cond_dict = {CONT_COV_1: torch.zeros((4, 1), requires_grad=False)}
        p1, _ = build_potential(
            adata,
            cond_dict,
            ystar,
            n_samples=1,
            potential_seed=0,
            conditions_covariates=(CONT_COV_1,),
        )
        p4, _ = build_potential(
            adata,
            cond_dict,
            ystar,
            n_samples=4,
            potential_seed=0,
            conditions_covariates=(CONT_COV_1,),
        )
        assert p1(cond_dict).shape == p4(cond_dict).shape == (4,)


# ---------------------------------------------------------------------------
# With groups
# ---------------------------------------------------------------------------


class TestWithGroups:
    def test_single_group_column(self, adata, cond_dict_single, ystar):
        potential, _ = build_potential(
            adata,
            cond_dict_single,
            ystar,
            groups=("target",),
            groups_encoding={"target": "label"},
            conditions_covariates=(CONT_COV_1,),
        )
        out = potential(cond_dict_single)
        assert out.shape == (4,)

    def test_group_and_categorical_condition(self, adata, cond_dict_single, ystar):
        potential, _ = build_potential(
            adata,
            cond_dict_single,
            ystar,
            conditions={"drug": ("drugA", "drugB")},
            conditions_reps={"drug": "drug"},
            groups=("source_split",),
            groups_encoding={"source_split": "label"},
            conditions_covariates=(CONT_COV_1,),
        )
        out = potential(cond_dict_single)
        assert out.shape == (4,)

    def test_generate_from_noise_with_groups(self, adata, cond_dict_single, ystar):
        potential, _ = build_potential(
            adata,
            cond_dict_single,
            ystar,
            n_samples=4,
            groups=("target",),
            groups_encoding={"target": "label"},
            conditions_covariates=(CONT_COV_1,),
        )
        out = potential(cond_dict_single)
        assert out.shape == (4,)


# ---------------------------------------------------------------------------
# With controls
# ---------------------------------------------------------------------------


class TestWithControls:
    def test_control_values_dict_marks_sources(self, adata, cond_dict_single, ystar):
        potential, _ = build_potential(
            adata,
            cond_dict_single,
            ystar,
            conditions={"drug": ("drugA", "drugB")},
            conditions_reps={"drug": "drug"},
            groups=("source_split",),
            groups_encoding={"source_split": "label"},
            control_values_dict={"drug": "control"},
            conditions_covariates=(CONT_COV_1,),
        )
        out = potential(cond_dict_single)
        assert out.shape == (4,)

    def test_controls_with_generate_from_noise(self, adata, cond_dict_single, ystar):
        potential, _ = build_potential(
            adata,
            cond_dict_single,
            ystar,
            n_samples=4,
            conditions={"drug": ("drugA", "drugB")},
            conditions_reps={"drug": "drug"},
            groups=("source_split",),
            groups_encoding={"source_split": "label"},
            control_values_dict={"drug": "control"},
            conditions_covariates=(CONT_COV_1,),
        )
        out = potential(cond_dict_single)
        assert out.shape == (4,)

    def test_gradient_reaches_conditions_under_matching(self, adata, ystar):
        cond_dict = {CONT_COV_1: torch.randn(4, 1, requires_grad=True)}
        potential, _ = build_potential(
            adata,
            cond_dict,
            ystar,
            conditions={"drug": ("drugA", "drugB")},
            conditions_reps={"drug": "drug"},
            groups=("source_split",),
            groups_encoding={"source_split": "label"},
            control_values_dict={"drug": "control"},
            conditions_covariates=(CONT_COV_1,),
        )
        out = potential(cond_dict)
        out.sum().backward()
        assert cond_dict[CONT_COV_1].grad is not None


# ---------------------------------------------------------------------------
# Without target state (metadata-only prediction)
# ---------------------------------------------------------------------------


class TestWithoutTargetState:
    def test_forward_runs_without_state(self, adata, cond_dict_single, ystar):
        potential, _ = build_potential(
            adata,
            cond_dict_single,
            ystar,
            require_target_state=False,
            conditions_covariates=(CONT_COV_1,),
        )
        out = potential(cond_dict_single)
        assert out.shape == (4,)

    def test_gradient_reaches_conditions(self, adata, ystar):
        cond_dict = {CONT_COV_1: torch.randn(4, 1, requires_grad=True)}
        potential, _ = build_potential(
            adata,
            cond_dict,
            ystar,
            require_target_state=False,
            conditions_covariates=(CONT_COV_1,),
        )
        out = potential(cond_dict)
        out.sum().backward()
        assert cond_dict[CONT_COV_1].grad is not None

    def test_generate_from_noise_without_state(self, adata, cond_dict_single, ystar):
        potential, _ = build_potential(
            adata,
            cond_dict_single,
            ystar,
            n_samples=4,
            require_target_state=False,
            conditions_covariates=(CONT_COV_1,),
        )
        out = potential(cond_dict_single)
        assert out.shape == (4,)


# ---------------------------------------------------------------------------
# Parameter sweeps
# ---------------------------------------------------------------------------


class TestParameterSweeps:
    @pytest.mark.parametrize("N", [1, 2, 8, 32])
    def test_condition_counts(self, adata, ystar, N):
        cond_dict = {CONT_COV_1: torch.zeros((N, 1), requires_grad=True)}
        potential, _ = build_potential(
            adata,
            cond_dict,
            ystar,
            conditions_covariates=(CONT_COV_1,),
        )
        assert potential(cond_dict).shape == (N,)

    @pytest.mark.parametrize("batch_size", [8, 32, 128])
    def test_batch_size_does_not_affect_output(self, adata, cond_dict_single, ystar, batch_size):
        potential, _ = build_potential(
            adata,
            cond_dict_single,
            ystar,
            batch_size=batch_size,
            conditions_covariates=(CONT_COV_1,),
        )
        assert potential(cond_dict_single).shape == (4,)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_output(self, adata, cond_dict_single, ystar):
        p1, _ = build_potential(
            adata,
            cond_dict_single,
            ystar,
            potential_seed=7,
            n_samples=4,
            conditions_covariates=(CONT_COV_1,),
        )
        p2, _ = build_potential(
            adata,
            cond_dict_single,
            ystar,
            potential_seed=7,
            n_samples=4,
            conditions_covariates=(CONT_COV_1,),
        )
        assert torch.allclose(p1(cond_dict_single), p2(cond_dict_single))

    def test_different_seed_different_output(self, adata, cond_dict_single, ystar):
        p1, _ = build_potential(
            adata,
            cond_dict_single,
            ystar,
            potential_seed=1,
            n_samples=4,
            conditions_covariates=(CONT_COV_1,),
        )
        p2, _ = build_potential(
            adata,
            cond_dict_single,
            ystar,
            potential_seed=2,
            n_samples=4,
            conditions_covariates=(CONT_COV_1,),
        )
        assert not torch.allclose(p1(cond_dict_single), p2(cond_dict_single))

    def test_loader_cached_across_calls(self, adata, cond_dict_single, ystar):
        potential, _ = build_potential(
            adata,
            cond_dict_single,
            ystar,
            conditions_covariates=(CONT_COV_1,),
        )
        potential(cond_dict_single)
        potential(cond_dict_single)
        assert potential.predict_dl is potential.predict_dl


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestErrorPaths:
    def test_empty_condition_dict(self, adata, cond_dict_single, ystar):
        potential, _ = build_potential(
            adata,
            cond_dict_single,
            ystar,
            conditions_covariates=(CONT_COV_1,),
        )
        with pytest.raises(ValueError, match="empty condition dictionary"):
            potential({})

    def test_unknown_condition_key(self, adata, cond_dict_single, ystar):
        potential, _ = build_potential(
            adata,
            cond_dict_single,
            ystar,
            conditions_covariates=(CONT_COV_1,),
        )
        bad = {"nope": torch.zeros((4, 1))}
        with pytest.raises(ValueError, match="does not appear in the reference"):
            potential(bad)

    def test_wrong_trailing_dim(self, adata, cond_dict_single, ystar):
        potential, _ = build_potential(
            adata,
            cond_dict_single,
            ystar,
            conditions_covariates=(CONT_COV_1,),
        )
        bad = {CONT_COV_1: torch.zeros((4, 99))}
        with pytest.raises(ValueError, match="Shape mismatch for condition key"):
            potential(bad)

    def test_inconsistent_leading_dim(self, adata, cond_dict_pair, ystar):
        potential, _ = build_potential(
            adata,
            cond_dict_pair,
            ystar,
            conditions_covariates=(CONT_COV_1, CONT_COV_2),
        )
        bad = {
            CONT_COV_1: torch.zeros((4, 1)),
            CONT_COV_2: torch.zeros((3, 1)),
        }
        with pytest.raises(ValueError, match="wrong number of optimization samples"):
            potential(bad)

    def test_non_2d_condition(self, adata, cond_dict_single, ystar):
        potential, _ = build_potential(
            adata,
            cond_dict_single,
            ystar,
            conditions_covariates=(CONT_COV_1,),
        )
        with pytest.raises(ValueError, match="two-dimensional"):
            potential({CONT_COV_1: torch.zeros(4)})

    def test_no_predict_data_set(self, adata, cond_dict_single, ystar):
        dmod = build_datamodule(adata, conditions_covariates=(CONT_COV_1,))
        inferer = FakeInferer(ystar.numel())
        potential = MeanMSESurrogate(inferer, ystar, dmod)
        with pytest.raises(ValueError, match="no prediction data"):
            potential(cond_dict_single)


# ---------------------------------------------------------------------------
# Regression: sample_rep vs default .X
# ---------------------------------------------------------------------------


class TestStateRepresentations:
    def test_default_uses_dot_x(self, adata, cond_dict_single, ystar):
        potential, _ = build_potential(
            adata,
            cond_dict_single,
            ystar,
            sample_rep=None,
            conditions_covariates=(CONT_COV_1,),
        )
        assert potential(cond_dict_single).shape == (4,)

    def test_uses_named_obsm_representation(self, adata, cond_dict_single, ystar, repr_obsm_key, n_feats_obsm_repr):
        ystar_repr = torch.zeros(n_feats_obsm_repr)
        potential, _ = build_potential(
            adata,
            cond_dict_single,
            ystar_repr,
            sample_rep=repr_obsm_key,
            conditions_covariates=(CONT_COV_1,),
        )
        assert potential(cond_dict_single).shape == (4,)


# ---------------------------------------------------------------------------
# Regression: ystar/mask handling under the real schema
# ---------------------------------------------------------------------------


class TestYstarAndMask:
    def test_mask_narrows_response(self, adata, cond_dict_single, n_genes):
        mask = torch.zeros(n_genes, dtype=torch.bool)
        mask[:8] = True
        ystar = torch.zeros(n_genes)
        dmod = build_datamodule(adata, conditions_covariates=(CONT_COV_1,))
        dmod.set_predict_data(adata)
        inferer = FakeInferer(n_genes)
        potential = MeanMSESurrogate(inferer, ystar, dmod, mask=mask)
        out = potential(cond_dict_single)
        assert out.shape == (4,)

    def test_ystar_moves_with_module(self, adata, cond_dict_single, n_genes):
        potential, _ = build_potential(
            adata,
            cond_dict_single,
            torch.zeros(n_genes),
            conditions_covariates=(CONT_COV_1,),
        )
        assert potential.ystar.device == potential.mask.device
        assert potential.mask.dtype == torch.bool
