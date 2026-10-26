"""Tests for sckitflow._predict: batching, reassembly, and the helpers that connect them."""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("anndata")

import sckitflow._predict as predict_module
from sckitflow._predict import (
    _get_pred_obsm_dict,
    _get_pred_raw_samples,
    _get_pred_traj,
    _pred_obs_from_leaf,
    _predict_empty,
    _to_numpy,
    predict_adata,
    prediction_record,
    predictions_to_adata,
)

# ============================================================================
# Fakes
# ============================================================================


@dataclasses.dataclass(frozen=True)
class FakePredictionData:
    """A drop-in for sckitflow's PredictionData with the same three fields used here."""

    X: torch.Tensor
    raw_samples: torch.Tensor | None = None
    traj: torch.Tensor | None = None


def make_step_data(n_obs: int, n_features: int = 4, n_cond: int = 2, with_source: bool = True) -> dict:
    state = torch.arange(n_obs * n_features, dtype=torch.float32).reshape(n_obs, n_features)
    return {
        "target_state": state,
        "source_state": state + 1 if with_source else None,
        "target_condition_data": {"cond": torch.arange(n_obs * n_cond, dtype=torch.float32).reshape(n_obs, n_cond)},
        "target_response_data": {},
        "target_group_data": {"g": torch.ones(1, 3)},  # one row for the whole group, never split
        "n_obs": n_obs,
    }


class FakeLoader:
    def __init__(
        self,
        items,
        cond_cont_keys: tuple[str, ...] = (),
        resp_keys: tuple[str, ...] = (),
        group_cols: tuple[str, ...] = (),
    ):
        self.items = items
        self.cond_cont_keys = cond_cont_keys
        self.resp_keys = resp_keys
        self.group_cols = group_cols
        self.source_obs_names: dict = {}

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)


class FakeInference:
    """Records every predict() call and returns a simple PredictionData."""

    def __init__(self, n_features: int = 4, with_traj: bool = False, n_samples: int = 1):
        self.module = torch.nn.Linear(1, 1)
        self.module.train(False)
        self.n_features = n_features
        self.with_traj = with_traj
        self.n_samples = n_samples
        self.calls: list[dict] = []

    def predict(self, step_data, generator, **kwargs):
        n_obs = step_data["target_state"].shape[0]
        assert step_data["n_obs"] == n_obs and step_data["target_group_data"]["g"].shape[0] == 1
        self.calls.append({"n_obs": n_obs})
        X = step_data["target_state"].clone()
        raw_samples = None
        if self.n_samples > 1:
            raw_samples = X.unsqueeze(0).expand(self.n_samples, -1, -1).clone()
        traj = X.unsqueeze(1) if self.with_traj else None
        return FakePredictionData(X=X, raw_samples=raw_samples, traj=traj)


def fake_datamodule(loader, feature_names):
    dm = MagicMock()
    dm.dtype = torch.float32
    dm.data_dims = MagicMock(feature_names=list(feature_names))
    dm.dm.get_eval_loader.return_value = loader
    return dm


# ============================================================================
# _to_numpy
# ============================================================================


class TestToNumpy:
    def test_tensor(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        out = _to_numpy(t)
        np.testing.assert_array_equal(out, np.array([1.0, 2.0, 3.0]))

    def test_none(self):
        assert _to_numpy(None) is None

    def test_array_like(self):
        out = _to_numpy([1, 2, 3])
        np.testing.assert_array_equal(out, np.array([1, 2, 3]))


# ============================================================================
# _predict_empty
# ============================================================================


class TestPredictEmpty:
    def test_shape_and_var(self):
        dims = MagicMock(feature_names=["a", "b", "c"])
        adata = _predict_empty(dims, return_raw=False)
        assert adata.shape == (0, 3)
        assert list(adata.var_names) == ["a", "b", "c"]

    def test_return_raw(self):
        dims = MagicMock(feature_names=["a"])
        adata, raw = _predict_empty(dims, return_raw=True)
        assert adata.shape == (0, 1)
        assert raw is None


# ============================================================================
# _pred_obs_from_leaf
# ============================================================================


class TestPredObsFromLeaf:
    def test_single_row(self):
        pred = FakePredictionData(X=torch.zeros(4, 3))
        df = _pred_obs_from_leaf(("g",), ("ctrl",), pred)
        assert len(df) == 4
        assert list(df["g"]) == ["ctrl"] * 4

    def test_no_group_cols(self):
        pred = FakePredictionData(X=torch.zeros(2, 3))
        df = _pred_obs_from_leaf((), (), pred)
        assert len(df) == 2
        assert list(df.columns) == []

    def test_multiple_group_cols(self):
        pred = FakePredictionData(X=torch.zeros(2, 3))
        df = _pred_obs_from_leaf(("a", "b"), ("x", "y"), pred)
        assert list(df["a"]) == ["x", "x"]
        assert list(df["b"]) == ["y", "y"]


# ============================================================================
# _get_pred_traj / _get_pred_raw_samples
# ============================================================================


class TestGetPredTraj:
    def test_none(self):
        assert _get_pred_traj(FakePredictionData(X=torch.zeros(3, 2))) is None

    def test_2d_obs_first(self):
        pred = FakePredictionData(X=torch.zeros(3, 2), traj=torch.zeros(3, 5))
        out = _get_pred_traj(pred)
        assert out.shape == (3, 5)

    def test_3d_time_first(self):
        pred = FakePredictionData(X=torch.zeros(3, 2), traj=torch.zeros(5, 3, 2))
        out = _get_pred_traj(pred)
        assert out.shape == (3, 5, 2)

    def test_4d_iter_time_obs(self):
        pred = FakePredictionData(X=torch.zeros(3, 2), traj=torch.zeros(4, 5, 3, 2))
        out = _get_pred_traj(pred)
        assert out.shape == (3, 4, 5, 2)

    def test_incompatible_shape_raises(self):
        pred = FakePredictionData(X=torch.zeros(3, 2), traj=torch.zeros(7, 8, 9))
        with pytest.raises(ValueError, match="incompatible shape"):
            _get_pred_traj(pred)


class TestGetPredRawSamples:
    def test_none(self):
        pred = FakePredictionData(X=torch.zeros(3, 2))
        assert _get_pred_raw_samples(pred) is None

    def test_2d(self):
        pred = FakePredictionData(X=torch.zeros(3, 2), raw_samples=torch.zeros(3, 2))
        out = _get_pred_raw_samples(pred)
        assert out.shape == (3, 2)

    def test_3d_samples_first(self):
        # (n_samples, n_obs, n_features) -> transposed to (n_obs, n_samples, n_features)
        pred = FakePredictionData(X=torch.zeros(3, 2), raw_samples=torch.zeros(10, 3, 2))
        out = _get_pred_raw_samples(pred)
        assert out.shape == (3, 10, 2)

    def test_incompatible_shape_raises(self):
        pred = FakePredictionData(X=torch.zeros(3, 2), raw_samples=torch.zeros(7, 8, 9))
        with pytest.raises(ValueError, match="incompatible shape"):
            _get_pred_raw_samples(pred)


# ============================================================================
# _get_pred_obsm_dict
# ============================================================================


class TestGetPredObsmDict:
    def test_empty(self):
        pred = FakePredictionData(X=torch.zeros(2, 3))
        out = _get_pred_obsm_dict(
            {"target_condition_data": None, "target_response_data": None},
            pred,
            cont_keys=(),
        )
        assert out == {}

    def test_traj_and_samples(self):
        # raw_samples must be (n_samples, n_obs, n_features):
        # 5 samples, 2 obs, 3 features -> transposes to (2, 5, 3).
        pred = FakePredictionData(
            X=torch.zeros(2, 3),
            traj=torch.zeros(2, 4),
            raw_samples=torch.zeros(5, 2, 3),
        )
        out = _get_pred_obsm_dict(
            {"target_condition_data": None, "target_response_data": None},
            pred,
            cont_keys=(),
        )
        assert "trajectory" in out
        assert "raw_samples" in out
        assert out["raw_samples"].shape == (2, 5, 3)

    def test_cont_keys(self):
        pred = FakePredictionData(X=torch.zeros(2, 3))
        step_data = {
            "target_condition_data": {"x": torch.ones(2, 4)},
            "target_response_data": {"y": torch.zeros(2, 5)},
        }
        out = _get_pred_obsm_dict(step_data, pred, cont_keys=("x", "y", "missing"))
        assert set(out.keys()) == {"x", "y"}
        assert out["x"].shape == (2, 4)
        assert out["y"].shape == (2, 5)


# ============================================================================
# prediction_record + predictions_to_adata
# ============================================================================


class TestPredictionRecord:
    def test_shape_of_record(self):
        step_data = make_step_data(3)
        pred = FakePredictionData(X=torch.zeros(3, 4))
        loader = FakeLoader(
            items=[],
            cond_cont_keys=("cond",),
            resp_keys=(),
            group_cols=("g",),
        )
        record = prediction_record(loader, step_data, ("ctrl",), pred)
        assert record["preds_cls"] is FakePredictionData
        assert set(record["preds_fields"].keys()) == {"X", "raw_samples", "traj"}
        assert list(record["obs"]["g"]) == ["ctrl"] * 3
        assert "cond" in record["obsm"]


class TestPredictionsToAdata:
    def test_empty_records(self):
        dims = MagicMock(feature_names=["a", "b"])
        adata = predictions_to_adata(dims, records=[])
        assert adata.shape == (0, 2)

    def test_single_record(self):
        dims = MagicMock(feature_names=["a", "b", "c", "d"])
        record = {
            "preds_cls": FakePredictionData,
            "preds_fields": {"X": torch.zeros(3, 4), "raw_samples": None, "traj": None},
            "obs": pd.DataFrame({"g": ["x"] * 3}, index=pd.RangeIndex(3)),
            "obsm": {},
        }
        with patch.object(
            predict_module,
            "concatenate_predictions",
            side_effect=lambda preds: preds[0],
        ):
            adata = predictions_to_adata(dims, records=[record])
        assert adata.shape == (3, 4)
        assert list(adata.var_names) == ["a", "b", "c", "d"]


# ============================================================================
# predict_adata: the main entry point
# ============================================================================


class TestPredictAdataNoBatching:
    def test_single_group_single_call(self):
        step_data = make_step_data(20)
        loader = FakeLoader([(step_data, ("g",))])
        dm = fake_datamodule(loader, ["f0", "f1", "f2", "f3"])
        inf = FakeInference(n_features=4)

        with patch.object(predict_module, "predictions_to_adata", return_value=MagicMock()):
            predict_adata(dm, inf, MagicMock(), batch_size=None)

        assert [c["n_obs"] for c in inf.calls] == [20]

    def test_multiple_groups(self):
        loader = FakeLoader(
            [
                (make_step_data(5), ("g1",)),
                (make_step_data(7), ("g2",)),
                (make_step_data(3), ("g3",)),
            ]
        )
        dm = fake_datamodule(loader, ["f0", "f1", "f2", "f3"])
        inf = FakeInference(n_features=4)

        with patch.object(predict_module, "predictions_to_adata", return_value=MagicMock()):
            predict_adata(dm, inf, MagicMock(), batch_size=None)

        assert [c["n_obs"] for c in inf.calls] == [5, 7, 3]


class TestPredictAdataBatching:
    def test_even_split(self):
        step_data = make_step_data(200)
        loader = FakeLoader([(step_data, ("g",))])
        dm = fake_datamodule(loader, ["f0", "f1", "f2", "f3"])
        inf = FakeInference(n_features=4)

        with patch.object(predict_module, "predictions_to_adata", return_value=MagicMock()):
            predict_adata(dm, inf, MagicMock(), batch_size=100)

        assert [c["n_obs"] for c in inf.calls] == [100, 100]

    def test_uneven_split(self):
        step_data = make_step_data(250)
        loader = FakeLoader([(step_data, ("g",))])
        dm = fake_datamodule(loader, ["f0", "f1", "f2", "f3"])
        inf = FakeInference(n_features=4)

        with patch.object(predict_module, "predictions_to_adata", return_value=MagicMock()):
            predict_adata(dm, inf, MagicMock(), batch_size=100)

        assert [c["n_obs"] for c in inf.calls] == [100, 100, 50]

    def test_batch_size_equals_n_obs(self):
        step_data = make_step_data(50)
        loader = FakeLoader([(step_data, ("g",))])
        dm = fake_datamodule(loader, ["f0", "f1", "f2", "f3"])
        inf = FakeInference(n_features=4)

        with patch.object(predict_module, "predictions_to_adata", return_value=MagicMock()):
            predict_adata(dm, inf, MagicMock(), batch_size=50)

        assert [c["n_obs"] for c in inf.calls] == [50]

    def test_batch_size_larger_than_n_obs(self):
        step_data = make_step_data(20)
        loader = FakeLoader([(step_data, ("g",))])
        dm = fake_datamodule(loader, ["f0", "f1", "f2", "f3"])
        inf = FakeInference(n_features=4)

        with patch.object(predict_module, "predictions_to_adata", return_value=MagicMock()):
            predict_adata(dm, inf, MagicMock(), batch_size=1000)

        assert [c["n_obs"] for c in inf.calls] == [20]

    def test_multiple_groups_each_batched(self):
        loader = FakeLoader(
            [
                (make_step_data(250), ("g1",)),
                (make_step_data(120), ("g2",)),
            ]
        )
        dm = fake_datamodule(loader, ["f0", "f1", "f2", "f3"])
        inf = FakeInference(n_features=4)

        with patch.object(predict_module, "predictions_to_adata", return_value=MagicMock()):
            predict_adata(dm, inf, MagicMock(), batch_size=100)

        # group 1: 100, 100, 50; group 2: 100, 20
        assert [c["n_obs"] for c in inf.calls] == [100, 100, 50, 100, 20]

    def test_batching_preserves_total_obs_via_record(self):
        """Every input row is accounted for exactly once in the reassembled prediction."""
        n_obs = 250
        step_data = make_step_data(n_obs)
        loader = FakeLoader([(step_data, ("g",))])
        dm = fake_datamodule(loader, ["f0", "f1", "f2", "f3"])
        inf = FakeInference(n_features=4)

        captured: list[dict] = []

        def fake_record(loader, step_data, leaf, preds):
            captured.append({"leaf": leaf, "X_shape": tuple(preds.X.shape)})
            return {
                "preds_cls": FakePredictionData,
                "preds_fields": {"X": preds.X},
                "obs": None,
                "obsm": {},
            }

        with (
            patch.object(predict_module, "prediction_record", side_effect=fake_record),
            patch.object(predict_module, "predictions_to_adata", return_value=MagicMock()),
        ):
            predict_adata(dm, inf, MagicMock(), batch_size=100)

        assert len(captured) == 1
        assert captured[0]["X_shape"] == (n_obs, 4)

    def test_batched_prediction_values_reassemble(self):
        """The batched output is the concatenation of the per-batch X tensors, in order."""
        n_obs = 250
        step_data = make_step_data(n_obs)
        # Fill target_state with row index so we can verify order after reassembly.
        step_data["target_state"] = torch.arange(n_obs, dtype=torch.float32).unsqueeze(1).expand(n_obs, 4).clone()
        loader = FakeLoader([(step_data, ("g",))])
        dm = fake_datamodule(loader, ["f0", "f1", "f2", "f3"])
        inf = FakeInference(n_features=4)

        captured: list[torch.Tensor] = []

        def fake_record(loader, step_data, leaf, preds):
            captured.append(preds.X)
            return {
                "preds_cls": FakePredictionData,
                "preds_fields": {"X": preds.X},
                "obs": None,
                "obsm": {},
            }

        with (
            patch.object(predict_module, "prediction_record", side_effect=fake_record),
            patch.object(predict_module, "predictions_to_adata", return_value=MagicMock()),
        ):
            predict_adata(dm, inf, MagicMock(), batch_size=100)

        reassembled = captured[0]
        expected = torch.arange(n_obs, dtype=torch.float32).unsqueeze(1).expand(n_obs, 4)
        assert torch.allclose(reassembled, expected)

    def test_zero_batch_size_raises(self):
        dm = fake_datamodule(FakeLoader([]), ["f0"])
        inf = FakeInference()
        with pytest.raises(ValueError, match="batch_size must be positive"):
            predict_adata(dm, inf, MagicMock(), batch_size=0)

    def test_negative_batch_size_raises(self):
        dm = fake_datamodule(FakeLoader([]), ["f0"])
        inf = FakeInference()
        with pytest.raises(ValueError, match="batch_size must be positive"):
            predict_adata(dm, inf, MagicMock(), batch_size=-1)

    def test_empty_loader(self):
        loader = FakeLoader([])
        dm = fake_datamodule(loader, ["f0", "f1"])
        inf = FakeInference(n_features=2)

        with patch.object(predict_module, "predictions_to_adata", return_value=MagicMock()) as mock_pta:
            predict_adata(dm, inf, MagicMock(), batch_size=100)

        assert inf.calls == []
        mock_pta.assert_called_once()
        # empty records list
        assert mock_pta.call_args.args[1] == []

    def test_module_mode_restored(self):
        """`predict_adata` restores the original training mode of the module on exit."""
        loader = FakeLoader([(make_step_data(10), ("g",))])
        dm = fake_datamodule(loader, ["f0", "f1"])
        inf = FakeInference(n_features=2)
        inf.module.train(True)
        assert inf.module.training is True

        with patch.object(predict_module, "predictions_to_adata", return_value=MagicMock()):
            predict_adata(dm, inf, MagicMock(), batch_size=None)

        assert inf.module.training is True

    def test_module_mode_restored_on_exception(self):
        """If `predict` raises, the module mode is still restored before the error propagates."""
        loader = FakeLoader([(make_step_data(10), ("g",))])
        dm = fake_datamodule(loader, ["f0", "f1"])

        class RaisingInference(FakeInference):
            def predict(self, step_data, generator, **kwargs):
                raise RuntimeError("boom")

        inf = RaisingInference(n_features=2)
        inf.module.train(True)

        with pytest.raises(RuntimeError, match="boom"):
            predict_adata(dm, inf, MagicMock(), batch_size=100)

        assert inf.module.training is True
