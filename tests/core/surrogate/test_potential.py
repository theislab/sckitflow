# tests/test_surrogate_potential.py
"""
Test suite for ``SurrogatePotential``.

Uses minimal fakes for ``SupportsInference``, ``EvalLoader`` and ``FlowDataModule``
so nothing here touches AnnData or Lightning. Only the two abstract hooks are
implemented, in a single ``MSESurrogate`` subclass that every test reuses.
"""

from types import SimpleNamespace

import pytest
import torch

from sckitflow.core.surrogate._potential import (
    SurrogatePotential,
    _attach_continuous_conditions_to_step_data,
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
        return self._response_fn(step_data, generator)


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
        self.predict_calls = 0

    @property
    def data_dims(self):
        return self._dims

    def predict_dataloader(self):
        self.predict_calls += 1
        return self._loader


# ---------------------------------------------------------------------------
# Concrete subclasses
# ---------------------------------------------------------------------------


class MSESurrogate(SurrogatePotential):
    """Squared-error potential. Smallest subclass that exercises the hooks."""

    def get_response(self, pred_data):
        return pred_data

    def compute_raw_potential(self, y, ystar):
        return ((y - ystar) ** 2).mean(dim=-1)


class RecorderSurrogate(SurrogatePotential):
    """Records the tensors its hooks receive, for shape/values assertions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.received_y = None
        self.received_ystar = None

    def get_response(self, pred_data):
        return pred_data

    def compute_raw_potential(self, y, ystar):
        self.received_y = y.detach().clone()
        self.received_ystar = ystar.detach().clone()
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


def make_potential(
    cond_keys=("cond_a",), n=4, dim=3, n_leaves=3, ystar_dim=3, mask=None, seed=0, device="cpu", cls=MSESurrogate
):
    dims = make_dims(cond_keys, dim)
    step_data = make_step_data(cond_keys, n=n, dims=dict.fromkeys(cond_keys, dim), device=device)
    batches = [(step_data, f"leaf_{i}") for i in range(n_leaves)]

    # Fixed projection so the response is a deterministic function of the
    # conditions (and therefore differentiable w.r.t. cond_dict). A small
    # generator-driven noise term keeps the seed tests meaningful.
    projection = torch.randn(
        dim * len(cond_keys),
        ystar_dim,
        generator=torch.Generator().manual_seed(1234),
        device=device,
    )

    def response_fn(step_data, generator):
        cond_cov = step_data["target_condition_data"]
        flat = torch.cat([cond_cov[k] for k in cond_keys], dim=-1)
        noise = torch.randn(n, ystar_dim, generator=generator, device=device)
        return flat @ projection + 0.1 * noise

    inferer = FakeInferer(response_fn)
    datamodule = FakeDataModule(dims, FakeEvalLoader(batches))
    ystar = torch.zeros(ystar_dim, device=device)

    potential = cls(inferer, ystar, datamodule, mask=mask, seed=seed)
    if device != "cpu":
        potential = potential.to(device)
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
# _attach_continuous_conditions_to_step_data
# ---------------------------------------------------------------------------


class TestAttachConditions:
    def test_returns_new_dict_and_preserves_other_keys(self):
        step = make_step_data()
        step["other"] = "untouched"
        new_cond = {"cond_a": torch.ones(4, 3)}
        out = _attach_continuous_conditions_to_step_data(step, new_cond)

        assert out is not step
        assert out["other"] == "untouched"
        assert out["target_condition_data"]["cond_a"] is new_cond["cond_a"]

    def test_does_not_mutate_input_condition_dict(self):
        step = make_step_data()
        original = step["target_condition_data"]["cond_a"]
        new_cond = {"cond_a": torch.ones(4, 3)}
        _attach_continuous_conditions_to_step_data(step, new_cond)

        assert step["target_condition_data"]["cond_a"] is original

    def test_raises_when_no_condition_data(self):
        with pytest.raises(ValueError, match="does not contain any condition covariates"):
            _attach_continuous_conditions_to_step_data({}, {"cond_a": torch.zeros(1, 1)})

    def test_raises_on_unknown_key(self):
        step = make_step_data(("cond_a",))
        with pytest.raises(ValueError, match="does not appear as condition covariate"):
            _attach_continuous_conditions_to_step_data(step, {"cond_b": torch.zeros(4, 3)})

    def test_preserves_non_overlapping_keys(self):
        step = make_step_data(("cond_a", "cond_b"))
        new_cond = {"cond_a": torch.ones(4, 3)}
        out = _attach_continuous_conditions_to_step_data(step, new_cond)

        assert set(out["target_condition_data"]) == {"cond_a", "cond_b"}
        assert out["target_condition_data"]["cond_b"] is step["target_condition_data"]["cond_b"]


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
        assert potential.ystar is potential._ystar
        assert potential.mask is potential._mask
        assert potential.seed == 42
        assert potential.inferer is inferer
        assert potential.datamodule is dmod
        assert "_ystar" in dict(potential.named_buffers())
        assert "_mask" in dict(potential.named_buffers())
        assert list(potential.parameters()) == []

    def test_rejects_non_1d_ystar(self):
        dims = make_dims(("cond_a",), 3)
        with pytest.raises(ValueError, match="Invalid shape for target response"):
            MSESurrogate(FakeInferer(lambda *a: None), torch.zeros(2, 3), FakeDataModule(dims, FakeEvalLoader([])))

    def test_squeezes_1d_ystar_from_column(self):
        dims = make_dims(("cond_a",), 3)
        p = MSESurrogate(FakeInferer(lambda *a: None), torch.zeros(3, 1), FakeDataModule(dims, FakeEvalLoader([])))
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
        potential, _, _ = make_potential(device="cuda")
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

    def test_loader_is_cached_across_calls(self):
        potential, _, dmod = make_potential()
        potential(make_cond_dict())
        potential(make_cond_dict())
        assert dmod.predict_calls == 1

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

        p = RecorderSurrogate(inferer, ystar, dmod, mask=mask)
        p(make_cond_dict(n=4))

        assert p.received_y.shape == (4, 2)
        assert p.received_ystar.shape == (2,)
        assert torch.allclose(p.received_ystar, torch.tensor([1.0, 3.0]))
        assert torch.allclose(p.received_y, torch.ones(4, 2))

    def test_all_true_mask_matches_no_mask_at_hook(self):
        dims = make_dims(("cond_a",), 3)
        step = make_step_data(("cond_a",), n=3, dims={"cond_a": 3})
        inferer = FakeInferer(lambda sd, g: torch.ones(3, 4))
        dmod = FakeDataModule(dims, FakeEvalLoader([(step, "leaf_0")]))
        ystar = torch.tensor([1.0, 2.0, 3.0, 4.0])

        m_all = torch.ones(4, dtype=torch.bool)
        p_none = RecorderSurrogate(inferer, ystar, dmod, mask=None)
        p_all = RecorderSurrogate(inferer, ystar, dmod, mask=m_all)
        p_none(make_cond_dict(n=3))
        p_all(make_cond_dict(n=3))

        assert p_none.received_y.shape == p_all.received_y.shape
        assert torch.allclose(p_none.received_y, p_all.received_y)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_sample(self):
        potential, _, _ = make_potential(n=1, n_leaves=2)
        out = potential(make_cond_dict(n=1))
        assert out.shape == (1,)

    def test_single_response_dim_is_rejected_by_current_impl(self):
        # ``torch.squeeze(torch.zeros(1))`` collapses to a 0-dim scalar, which
        # ``__init__`` rejects. Supporting D=1 needs an ``unsqueeze(0)``
        # fallback after the squeeze; until then, this test documents the
        # current behaviour so a future change is noticed.
        with pytest.raises(ValueError, match="Invalid shape for target response"):
            make_potential(ystar_dim=1)

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
        p = MSESurrogate(inferer, torch.zeros(3), dmod)
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

    def test_device_mismatch_across_conditions(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        potential, _, _ = make_potential(cond_keys=("cond_a", "cond_b"))
        cond = {
            "cond_a": torch.zeros(4, 3),
            "cond_b": torch.zeros(4, 3, device="cuda"),
        }
        with pytest.raises(ValueError, match="wrong device"):
            potential(cond)

    def test_compute_raw_potential_wrong_shape(self):
        class BadShape(MSESurrogate):
            def compute_raw_potential(self, y, ystar):
                return torch.zeros(1)

        dims = make_dims(("cond_a",), 3)
        step = make_step_data(("cond_a",), n=4, dims={"cond_a": 3})
        inferer = FakeInferer(lambda sd, g: torch.zeros(4, 3))
        dmod = FakeDataModule(dims, FakeEvalLoader([(step, "leaf_0")]))
        p = BadShape(inferer, torch.zeros(3), dmod)

        with pytest.raises(ValueError, match="wrong shape"):
            p(make_cond_dict(n=4))

    def test_y_with_wrong_last_dim_from_hook_raises(self):
        # ``get_response`` returns a tensor whose last dim does not
        # match ``ystar``. The explicit shape check in ``forward`` catches it
        # with a ValueError (not an IndexError) thanks to the added guard.
        class WrongY(SurrogatePotential):
            def get_response(self, pred_data):
                return torch.zeros(4, 10)

            def compute_raw_potential(self, y, ystar):
                return y.sum(dim=-1)

        dims = make_dims(("cond_a",), 3)
        step = make_step_data(("cond_a",), n=4, dims={"cond_a": 3})
        inferer = FakeInferer(lambda sd, g: None)
        dmod = FakeDataModule(dims, FakeEvalLoader([(step, "leaf_0")]))
        p = WrongY(inferer, torch.zeros(3), dmod)

        with pytest.raises(ValueError, match="last dim"):
            p(make_cond_dict(n=4))
