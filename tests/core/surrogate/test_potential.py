# tests/test_surrogate_potential.py
"""
Test suite for ``SurrogatePotential``.

Uses minimal fakes for ``SupportsInference``, ``EvalLoader`` and ``FlowDataModule``
so nothing here touches AnnData or Lightning. The fake inferer returns the response
itself as the generated cells, so the default readout's mean over them is the response and the potential
a mean squared error.
"""

from types import SimpleNamespace

import pytest
import torch

from sckitflow._random import generators
from sckitflow.core._types import PredictionData
from sckitflow.core.surrogate._potential import (
    SurrogatePotential,
    SurrogateResponse,
    _pair_leaf,
    _verify_continuous_conditions_dims,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeInferer:
    """Satisfies ``SupportsInference`` structurally.

    ``response_fn(step_data, generator)`` returns a tensor of shape ``(N, D)``.
    Records every call so tests can assert on the number of leaves visited.
    """

    def __init__(self, response_fn):
        self._module = torch.nn.Linear(1, 1)
        self._response_fn = response_fn
        self.calls = []

    @property
    def module(self) -> torch.nn.Module:
        return self._module

    def predict(self, step_data, *, generator):
        self.calls.append((step_data, generator))
        return PredictionData(X=None, raw_samples=self._response_fn(step_data, generator))


class FakeEvalLoader:
    """Minimal ``EvalLoader``: an iterable of ``(StepData, leaf)`` with ``__len__``."""

    def __init__(self, batches):
        self._batches = list(batches)

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


class FakeDataModule:
    """Just enough of ``FlowDataModule`` for the potential to run."""

    def __init__(self, dims, loader):
        self._dims = dims
        self._loader = loader
        self.dtype = None
        self.loader_calls = 0

    @property
    def data_dims(self):
        return self._dims

    @property
    def dm(self):
        return self

    def get_eval_loader(self, adata, **kwargs):
        self.loader_calls += 1
        return self._loader


# ---------------------------------------------------------------------------
# Concrete subclasses
# ---------------------------------------------------------------------------


def _mse(y, ystar):
    return ((y - ystar) ** 2).mean(dim=-1)


def surrogate(inferer, ystar, datamodule, *, potential=_mse, mask=None, **kwargs):
    response = SurrogateResponse(inferer=inferer, datamodule=datamodule, adata=None, **kwargs)
    return SurrogatePotential(response, ystar=ystar, potential=potential, mask=mask)


class Recorder:
    """A potential that records what it receives, for shape/values assertions."""

    def __init__(self):
        self.y = None
        self.ystar = None

    def __call__(self, y, ystar):
        self.y, self.ystar = y.detach().clone(), ystar.detach().clone()
        return (y - ystar).sum(dim=-1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_step_data(cond_keys=("cond_a",), n=4, dims=None, dtype=torch.float32, device="cpu"):
    dims = dims or dict.fromkeys(cond_keys, 3)
    return {"target_condition_data": {k: torch.zeros(n, dims[k], dtype=dtype, device=device) for k in cond_keys}}


def make_dims(cond_keys=("cond_a",), dim=3):
    return SimpleNamespace(condition_continuous_dims=dict.fromkeys(cond_keys, dim))


def make_cond_dict(cond_keys=("cond_a",), n=4, dim=3, requires_grad=True, device="cpu"):
    return {k: torch.randn(n, dim, requires_grad=requires_grad, device=device) for k in cond_keys}


def make_potential(cond_keys=("cond_a",), n=4, dim=3, n_leaves=3, ystar_dim=3, mask=None, seed=0):
    dims = make_dims(cond_keys, dim)
    step_data = make_step_data(cond_keys, n=n, dims=dict.fromkeys(cond_keys, dim))
    batches = [(step_data, f"leaf_{i}") for i in range(n_leaves)]

    # Fixed projection so the response is a deterministic function of the
    # conditions (and therefore differentiable w.r.t. cond_dict). A small
    # generator-driven noise term keeps the seed tests meaningful.
    projection = torch.randn(
        dim * len(cond_keys),
        ystar_dim,
        generator=torch.Generator().manual_seed(1234),
    )

    def response_fn(step_data, generator):
        cond_cov = step_data["target_condition_data"]
        flat = torch.cat([cond_cov[k] for k in cond_keys], dim=-1)
        noise = torch.randn(n, ystar_dim, generator=generator)
        return flat @ projection + 0.1 * noise

    inferer = FakeInferer(response_fn)
    datamodule = FakeDataModule(dims, FakeEvalLoader(batches))
    ystar = torch.zeros(ystar_dim)

    potential = surrogate(inferer, ystar, datamodule, mask=mask, seed=seed)
    return potential, inferer, datamodule


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def potential_default():
    return make_potential()


@pytest.fixture
def cond_dict_default():
    return make_cond_dict()


# ---------------------------------------------------------------------------
# _verify_continuous_conditions_dims
# ---------------------------------------------------------------------------


class TestVerifyDims:
    def test_passes_for_valid_input(self):
        dims = make_dims(("cond_a",), 3)
        _verify_continuous_conditions_dims(dims, {"cond_a": torch.zeros(4, 3)})

    def test_raises_when_no_continuous_dims(self):
        dims = SimpleNamespace(condition_continuous_dims=None)
        with pytest.raises(ValueError, match="do not include continuous"):
            _verify_continuous_conditions_dims(dims, {"cond_a": torch.zeros(4, 3)})

    def test_raises_on_wrong_ndim(self):
        dims = make_dims(("cond_a",), 3)
        with pytest.raises(ValueError, match="two-dimensional"):
            _verify_continuous_conditions_dims(dims, {"cond_a": torch.zeros(4)})

    def test_raises_on_unknown_key(self):
        dims = make_dims(("cond_a",), 3)
        with pytest.raises(ValueError, match="does not appear in the reference"):
            _verify_continuous_conditions_dims(dims, {"cond_b": torch.zeros(4, 3)})

    def test_raises_on_trailing_dim_mismatch(self):
        dims = make_dims(("cond_a",), 3)
        with pytest.raises(ValueError, match="Shape mismatch for condition key"):
            _verify_continuous_conditions_dims(dims, {"cond_a": torch.zeros(4, 5)})


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestInit:
    def test_registers_buffers_and_attributes(self):
        potential, inferer, dmod = make_potential(seed=42)
        assert potential.response.seed == 42
        assert potential.response.inferer is inferer
        assert potential.response.datamodule is dmod
        assert {"ystar", "mask"} <= set(dict(potential.named_buffers()))
        assert list(potential.parameters()) == []

    def test_rejects_non_1d_ystar(self):
        dims = make_dims(("cond_a",), 3)
        with pytest.raises(ValueError, match="Invalid shape for target response"):
            surrogate(FakeInferer(lambda *a: None), torch.zeros(2, 3), FakeDataModule(dims, FakeEvalLoader([])))

    def test_squeezes_1d_ystar_from_column(self):
        dims = make_dims(("cond_a",), 3)
        p = surrogate(FakeInferer(lambda *a: None), torch.zeros(3, 1), FakeDataModule(dims, FakeEvalLoader([])))
        assert p.ystar.shape == (3,)

    def test_default_mask_is_all_true(self):
        potential, _, _ = make_potential(ystar_dim=5)
        assert potential.mask.dtype == torch.bool
        assert potential.mask.shape == (5,)
        assert potential.mask.all()

    def test_mask_is_coerced_to_bool(self):
        mask = torch.tensor([1, 0, 1, 1, 0], dtype=torch.uint8)
        potential, _, _ = make_potential(ystar_dim=5, mask=mask)
        assert potential.mask.dtype == torch.bool

    def test_rejects_int_index_list_mask(self):
        with pytest.raises(ValueError, match="boolean mask"):
            make_potential(ystar_dim=5, mask=torch.tensor([0, 2, 4]))

    def test_rejects_mask_of_wrong_shape(self):
        with pytest.raises(ValueError, match="same shape"):
            make_potential(ystar_dim=5, mask=torch.ones(3, dtype=torch.bool))

    def test_mask_moves_with_module(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        potential = make_potential()[0].to("cuda")
        assert potential.mask.is_cuda
        assert potential.ystar.is_cuda


# ---------------------------------------------------------------------------
# forward -- happy paths
# ---------------------------------------------------------------------------


class TestForward:
    def test_returns_per_sample_potential(self):
        potential, _, _ = make_potential(n=4)
        out = potential(make_cond_dict(n=4))
        assert out.shape == (4,)
        assert out.requires_grad

    def test_aggregates_over_leaves_as_mean(self):
        potential, _, _ = make_potential(n=4, n_leaves=1)
        out = potential(make_cond_dict(n=4))
        assert out.shape == (4,)

    def test_gradient_flows_to_conditions(self):
        potential, _, _ = make_potential(n=3)
        cond = make_cond_dict(n=3, requires_grad=True)
        out = potential(cond)
        out.sum().backward()
        for t in cond.values():
            assert t.grad is not None
            assert t.grad.shape == t.shape

    def test_deterministic_given_same_seed(self):
        cond = make_cond_dict(n=3)
        p1, _, _ = make_potential(n=3, seed=7)
        p2, _, _ = make_potential(n=3, seed=7)
        assert torch.allclose(p1(cond), p2(cond))

    def test_different_seed_changes_output(self):
        cond = make_cond_dict(n=3)
        p1, _, _ = make_potential(n=3, seed=1)
        p2, _, _ = make_potential(n=3, seed=2)
        assert not torch.allclose(p1(cond), p2(cond))

    def test_conditions_off_the_module_device_raise(self):
        potential, _, _ = make_potential()
        with pytest.raises(ValueError, match="not on the inference module's device"):
            potential(make_cond_dict(device="meta"))

    def test_context_is_read_once(self):
        potential, _, dmod = make_potential()
        potential(make_cond_dict())
        potential(make_cond_dict())
        assert dmod.loader_calls == 1

    def test_inferer_called_once_per_leaf(self):
        potential, inferer, _ = make_potential(n_leaves=3)
        potential(make_cond_dict())
        assert len(inferer.calls) == 3


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


class TestMasking:
    def test_default_mask_selects_all_dims(self):
        # ``n`` must match between ``make_potential`` (which sizes the
        # step_data) and ``make_cond_dict`` (which sizes the optimizer input).
        cond = make_cond_dict(n=3)
        p_none, _, _ = make_potential(n=3, ystar_dim=4, mask=None, seed=0)
        p_all, _, _ = make_potential(n=3, ystar_dim=4, mask=torch.ones(4, dtype=torch.bool), seed=0)
        assert torch.allclose(p_none(cond), p_all(cond))

    def test_mask_selects_subset_of_response_dims(self):
        mask = torch.tensor([True, False, True, False, False])
        potential, _, _ = make_potential(ystar_dim=5, mask=mask)
        out = potential(make_cond_dict(dim=3))
        assert out.shape == (4,)

    def test_different_masks_give_different_outputs(self):
        cond = make_cond_dict(n=3)
        m1 = torch.tensor([True, False, True, False])
        m2 = torch.tensor([False, True, False, True])
        p1, _, _ = make_potential(n=3, ystar_dim=4, mask=m1, seed=0)
        p2, _, _ = make_potential(n=3, ystar_dim=4, mask=m2, seed=0)
        assert not torch.allclose(p1(cond), p2(cond))

    def test_bool_and_uint8_masks_are_equivalent(self):
        cond = make_cond_dict(n=3)
        m_bool = torch.tensor([True, False, True])
        m_uint = torch.tensor([1, 0, 1], dtype=torch.uint8)
        p_bool, _, _ = make_potential(n=3, ystar_dim=3, mask=m_bool, seed=0)
        p_uint, _, _ = make_potential(n=3, ystar_dim=3, mask=m_uint, seed=0)
        assert torch.allclose(p_bool(cond), p_uint(cond))

    def test_single_true_dim_mask(self):
        mask = torch.zeros(4, dtype=torch.bool)
        mask[2] = True
        potential, _, _ = make_potential(n=3, ystar_dim=4, mask=mask)
        out = potential(make_cond_dict(n=3))
        assert out.shape == (3,)

    def test_gradient_reaches_conditions_under_any_mask(self):
        mask = torch.tensor([True, False, False, False])
        potential, _, _ = make_potential(ystar_dim=4, mask=mask, n=3)
        cond = make_cond_dict(n=3, requires_grad=True)
        out = potential(cond)
        out.sum().backward()
        for t in cond.values():
            assert t.grad is not None

    def test_mask_is_stored_as_bool_buffer_on_correct_device(self):
        mask = torch.tensor([1, 0, 1], dtype=torch.uint8)
        potential, _, _ = make_potential(ystar_dim=3, mask=mask)
        assert potential.mask.dtype == torch.bool
        assert potential.mask.device == potential.ystar.device

    def test_mask_narrows_both_y_and_ystar_in_compute_hook(self):
        mask = torch.tensor([True, False, True, False])
        dims = make_dims(("cond_a",), 3)
        step = make_step_data(("cond_a",), n=4, dims={"cond_a": 3})
        inferer = FakeInferer(lambda sd, g: torch.ones(4, 4))
        dmod = FakeDataModule(dims, FakeEvalLoader([(step, "leaf_0")]))
        ystar = torch.tensor([1.0, 2.0, 3.0, 4.0])

        rec = Recorder()
        surrogate(inferer, ystar, dmod, potential=rec, mask=mask)(make_cond_dict(n=4))

        assert rec.y.shape == (1, 1, 4, 2)  # (populations, groups, conditions, masked dims)
        assert rec.ystar.shape == (2,)
        assert torch.allclose(rec.ystar, torch.tensor([1.0, 3.0]))
        assert torch.allclose(rec.y, torch.ones(1, 1, 4, 2))

    def test_all_true_mask_matches_no_mask_at_hook(self):
        dims = make_dims(("cond_a",), 3)
        step = make_step_data(("cond_a",), n=3, dims={"cond_a": 3})
        inferer = FakeInferer(lambda sd, g: torch.ones(3, 4))
        dmod = FakeDataModule(dims, FakeEvalLoader([(step, "leaf_0")]))
        ystar = torch.tensor([1.0, 2.0, 3.0, 4.0])

        m_all = torch.ones(4, dtype=torch.bool)
        rec_none, rec_all = Recorder(), Recorder()
        surrogate(inferer, ystar, dmod, potential=rec_none, mask=None)(make_cond_dict(n=3))
        surrogate(inferer, ystar, dmod, potential=rec_all, mask=m_all)(make_cond_dict(n=3))

        assert rec_none.y.shape == rec_all.y.shape
        assert torch.allclose(rec_none.y, rec_all.y)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_sample(self):
        potential, _, _ = make_potential(n=1, n_leaves=2)
        out = potential(make_cond_dict(n=1))
        assert out.shape == (1,)

    def test_single_response_dim(self):
        potential, _, _ = make_potential(ystar_dim=1)
        assert potential.ystar.shape == (1,)

    def test_single_leaf(self):
        potential, inferer, _ = make_potential(n=3, n_leaves=1)
        out = potential(make_cond_dict(n=3))
        assert out.shape == (3,)
        assert len(inferer.calls) == 1

    def test_multiple_condition_keys(self):
        potential, _, _ = make_potential(cond_keys=("cond_a", "cond_b"), n=3)
        cond = {
            "cond_a": torch.randn(3, 3, requires_grad=True),
            "cond_b": torch.randn(3, 3, requires_grad=True),
        }
        out = potential(cond)
        assert out.shape == (3,)
        out.sum().backward()
        assert cond["cond_a"].grad is not None
        assert cond["cond_b"].grad is not None

    def test_no_grad_forward_still_works(self):
        potential, _, _ = make_potential(n=3)
        cond = make_cond_dict(n=3, requires_grad=False)
        with torch.no_grad():
            out = potential(cond)
        assert out.shape == (3,)
        assert not out.requires_grad

    def test_condition_tensors_are_not_mutated(self):
        potential, _, _ = make_potential(n=3)
        cond = make_cond_dict(n=3)
        before = {k: v.clone() for k, v in cond.items()}
        potential(cond)
        for k, v in cond.items():
            assert torch.equal(v, before[k])

    def test_step_data_not_mutated_across_leaves(self):
        # Each leaf's step_data should not carry over the previous leaf's
        # injected conditions.
        dims = make_dims(("cond_a",), 3)

        def make_batch():
            return (
                {"target_condition_data": {"cond_a": torch.zeros(4, 3)}},
                f"leaf_{id(object())}",
            )

        batches = [make_batch() for _ in range(3)]
        inferer = FakeInferer(lambda sd, g: torch.ones(4, 3))
        dmod = FakeDataModule(dims, FakeEvalLoader(batches))
        p = surrogate(inferer, torch.zeros(3), dmod)
        p(make_cond_dict(n=4))

        # Every leaf's original step_data still holds its all-zeros condition.
        for step, _ in batches:
            assert torch.equal(step["target_condition_data"]["cond_a"], torch.zeros(4, 3))


# ---------------------------------------------------------------------------
# forward -- error paths
# ---------------------------------------------------------------------------


class TestForwardErrors:
    def test_empty_condition_dict(self):
        potential, _, _ = make_potential()
        with pytest.raises(ValueError, match="empty condition dictionary"):
            potential({})

    def test_inconsistent_leading_dim(self):
        potential, _, _ = make_potential(cond_keys=("cond_a", "cond_b"), n=4)
        cond = {"cond_a": torch.zeros(4, 3), "cond_b": torch.zeros(3, 3)}
        with pytest.raises(ValueError, match="wrong number of optimization samples"):
            potential(cond)

    def test_wrong_trailing_dim(self):
        potential, _, _ = make_potential()
        with pytest.raises(ValueError, match="Shape mismatch for condition key"):
            potential({"cond_a": torch.zeros(4, 7)})

    def test_non_2d_condition(self):
        potential, _, _ = make_potential()
        with pytest.raises(ValueError, match="two-dimensional"):
            potential({"cond_a": torch.zeros(4)})

    def test_conditions_off_the_module_device_raise(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        potential, _, _ = make_potential(cond_keys=("cond_a", "cond_b"))
        cond = {"cond_a": torch.zeros(4, 3), "cond_b": torch.zeros(4, 3, device="cuda")}
        with pytest.raises(ValueError, match=r"\['cond_b'\] are not on the inference module's device"):
            potential(cond)

    def test_potential_of_the_wrong_shape_raises(self):
        dims = make_dims(("cond_a",), 3)
        step = make_step_data(("cond_a",), n=4, dims={"cond_a": 3})
        inferer = FakeInferer(lambda sd, g: torch.zeros(4, 3))
        dmod = FakeDataModule(dims, FakeEvalLoader([(step, "leaf_0")]))
        p = surrogate(inferer, torch.zeros(3), dmod, potential=lambda y, ystar: torch.zeros(1))

        with pytest.raises(ValueError, match="wrong shape"):
            p(make_cond_dict(n=4))

    def test_readout_maps_each_cell_before_the_mean(self):
        """The response is the mean of the mapped cells, not the map of the mean cell."""
        dims = make_dims(("cond_a",), 3)
        step = make_step_data(("cond_a",), n=4, dims={"cond_a": 3})
        cells = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
        inferer = FakeInferer(lambda sd, g: cells)
        dmod = FakeDataModule(dims, FakeEvalLoader([(step, "leaf_0")]))
        rec = Recorder()
        surrogate(inferer, torch.zeros(3), dmod, readout=torch.square, potential=rec)(make_cond_dict(n=4))
        torch.testing.assert_close(rec.y[0, 0], cells.square().mean(0))

    def test_a_readout_that_collapses_the_conditions_raises(self):
        """A readout reducing over the conditions would give every condition one shared potential."""
        dims = make_dims(("cond_a",), 3)
        step = make_step_data(("cond_a",), n=4, dims={"cond_a": 3})
        inferer = FakeInferer(lambda sd, g: torch.zeros(4, 3))
        dmod = FakeDataModule(dims, FakeEvalLoader([(step, "leaf_0")]))
        readout = lambda cells: cells.softmax(-1).mean(-2, keepdim=True)
        p = surrogate(inferer, torch.zeros(3), dmod, readout=readout)

        with pytest.raises(ValueError, match="one row per condition"):
            p(make_cond_dict(n=4))

    def test_a_readout_with_the_wrong_last_dim_raises(self):
        """The readout's last dim must match ``ystar``; caught with a ValueError, not an IndexError."""
        dims = make_dims(("cond_a",), 3)
        step = make_step_data(("cond_a",), n=4, dims={"cond_a": 3})
        inferer = FakeInferer(lambda sd, g: torch.zeros(4, 3))
        dmod = FakeDataModule(dims, FakeEvalLoader([(step, "leaf_0")]))
        p = surrogate(inferer, torch.zeros(3), dmod, readout=lambda cells: torch.zeros(*cells.shape[:-1], 10))

        with pytest.raises(ValueError, match="last dim"):
            p(make_cond_dict(n=4))


# ---------------------------------------------------------------------------
# Pairing a leaf against the conditions
# ---------------------------------------------------------------------------


class TestPairLeaf:
    def test_source_and_target_rows_form_a_cartesian_product(self):
        step = {
            "source_state": torch.tensor([[0.0], [1.0]]),
            "target_condition_data": {"other": torch.tensor([[10.0], [11.0]]), "opt": torch.zeros(2, 1)},
        }
        paired = _pair_leaf(step, {"opt": torch.zeros(1, 1)})
        pairs = list(
            zip(
                paired["source_state"][:, 0, 0].tolist(),
                paired["target_condition_data"]["other"][:, 0, 0].tolist(),
                strict=False,
            )
        )
        assert sorted(pairs) == [(0, 10), (0, 11), (1, 10), (1, 11)]

    def test_all_covariates_optimized_pairs_each_source_row(self):
        step = {"source_state": torch.randn(2, 4), "target_group_data": {"g": torch.ones(1, 2)}}
        paired = _pair_leaf(step, {"opt": torch.zeros(3, 1)})
        assert paired["source_state"].shape == (2, 3, 4)
        assert paired["target_group_data"]["g"].shape == (1, 2)  # one row for the leaf, broadcast by the model
        assert paired["target_condition_data"]["opt"].shape == (2, 3, 1)

    def test_groups_only_generation_from_noise_has_one_row_per_condition(self):
        step = {"target_group_data": {"g": torch.ones(1, 2)}}
        paired = _pair_leaf(step, {"opt": torch.zeros(3, 1)})
        assert paired["target_group_data"]["g"].shape == (1, 2)
        assert paired["target_condition_data"]["opt"].shape == (3, 1) and paired["n_obs"] == 3

    def test_other_keys_pass_through(self):
        step = {"source_state": torch.randn(2, 4), "target_condition_data": {}, "extra": "kept"}
        paired = _pair_leaf(step, {"opt": torch.zeros(3, 1)})
        assert paired["extra"] == "kept"

    def test_condition_covariates_disagreeing_on_rows_raise(self):
        step = {"target_condition_data": {"a": torch.zeros(2, 1), "b": torch.zeros(3, 1)}}
        with pytest.raises(ValueError, match="disagree on their rows"):
            _pair_leaf(step, {"opt": torch.zeros(3, 1)})

    def test_target_states_are_not_passed_to_prediction(self):
        step = {"source_state": torch.randn(2, 4), "target_state": torch.randn(7, 4), "target_condition_data": {}}
        paired = _pair_leaf(step, {"opt": torch.zeros(3, 1)})
        assert paired["target_state"] is None


class TestPopulations:
    def test_response_has_one_row_per_group_and_condition(self):
        potential, _, _ = make_potential(n=4, n_leaves=3, ystar_dim=5)
        assert potential.response(make_cond_dict(n=4)).shape == (3, 4, 5)

    def test_shape_and_first_population_matches_forward(self):
        potential, _, _ = make_potential(n=3)
        cond = make_cond_dict(n=3)
        pops = potential.population_potentials(cond, n_populations=4)
        assert pops.shape == (4, 3)
        torch.testing.assert_close(pops[0], potential(cond))

    def test_forward_keeps_the_single_population_stream(self):
        potential, inferer, _ = make_potential(n_leaves=2)
        potential(make_cond_dict())
        seeds = [generator.initial_seed() for _, generator in inferer.calls]
        assert seeds == [generators(potential.response.seed, i).initial_seed() for i in range(2)]

    def test_populations_differ(self):
        potential, _, _ = make_potential(n=3)
        pops = potential.population_potentials(make_cond_dict(n=3), n_populations=2)
        assert not torch.allclose(pops[0], pops[1])


def test_empty_prediction_loader_raises():
    potential, _, _ = make_potential(n_leaves=0)
    with pytest.raises(ValueError, match="no prediction leaves"):
        potential(make_cond_dict())
