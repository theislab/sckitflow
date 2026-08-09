"""Unit tests for :class:`sckitflow.data.splitters.CombinationSplitter`.

The policy under test: split unique ``group_keys`` combinations into train/test, hold out per
``always_train_keys`` stratum while always leaving >=1 combination of each stratum in train, and
label controls (never split) with ``control_label``.
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from sckitflow.data.splitters import CombinationSplitter

CELLS_PER_COMBO = 3


def _make_adata() -> AnnData:
    """cell_line x drug grid. A and B carry 6 perturbed drugs each; C carries only 1. All carry control."""
    combos: list[tuple[str, str]] = []
    for line, perturbed in {"A": 6, "B": 6, "C": 1}.items():
        combos.append((line, "control"))
        combos.extend((line, f"d{i}") for i in range(perturbed))
    rows = [combo for combo in combos for _ in range(CELLS_PER_COMBO)]
    obs = pd.DataFrame(rows, columns=["cell_line", "drug"]).astype("category")
    ad = AnnData(X=np.zeros((len(obs), 2), dtype=np.float32), obs=obs)
    return ad


def _splitter(**overrides) -> CombinationSplitter:
    kwargs = {
        "group_keys": ["cell_line", "drug"],
        "always_train_keys": ["cell_line"],
        "control_key": "drug",
        "control_value": "control",
        "test_fraction": 0.2,
        "seed": 0,
    }
    kwargs.update(overrides)
    return CombinationSplitter(**kwargs)


def _labelled(adata: AnnData, splitter: CombinationSplitter) -> pd.DataFrame:
    """Return obs with the split label column joined on, for convenient grouping."""
    labels = splitter.assign(adata)
    return adata.obs.assign(split=labels.to_numpy()).astype({"cell_line": str, "drug": str})


class TestCombinationSplitter:
    def test_controls_labelled_and_never_split(self):
        obs = _labelled(_make_adata(), _splitter())
        is_ctrl = obs["drug"] == "control"
        # every control row is labelled `control`; no non-control row is
        assert (obs.loc[is_ctrl, "split"] == "control").all()
        assert (obs.loc[~is_ctrl, "split"] != "control").all()

    def test_every_always_train_value_keeps_a_train_combo(self):
        """Each cell_line (the always_train stratum) must retain >=1 combination in train."""
        obs = _labelled(_make_adata(), _splitter(test_fraction=0.5))
        perturbed = obs[obs["drug"] != "control"]
        for cell_line, grp in perturbed.groupby("cell_line"):
            train_drugs = grp.loc[grp["split"] == "train", "drug"].nunique()
            assert train_drugs >= 1, f"{cell_line} was entirely held out of train"

    def test_single_combination_stratum_stays_in_train(self):
        """A stratum with one perturbed combination is never held out, even at a high test_fraction."""
        obs = _labelled(_make_adata(), _splitter(test_fraction=0.9))
        c_perturbed = obs[(obs["cell_line"] == "C") & (obs["drug"] != "control")]
        assert (c_perturbed["split"] == "train").all()

    def test_labels_are_only_train_test_control(self):
        obs = _labelled(_make_adata(), _splitter())
        assert set(obs["split"].unique()) <= {"train", "test", "control"}

    def test_holds_out_something_when_fraction_allows(self):
        """With enough combinations per stratum, some are actually held out to test."""
        obs = _labelled(_make_adata(), _splitter(test_fraction=0.5))
        assert (obs["split"] == "test").any()

    def test_zero_fraction_holds_out_nothing(self):
        labels = _splitter(test_fraction=0.0).assign(_make_adata())
        assert (labels != "test").all()

    def test_deterministic_given_seed(self):
        adata = _make_adata()
        a = _splitter(seed=0).assign(adata)
        b = _splitter(seed=0).assign(adata)
        pd.testing.assert_series_equal(a, b)

    def test_no_control_key_labels_no_controls(self):
        labels = _splitter(control_key=None).assign(_make_adata())
        assert (labels != "control").all()

    def test_split_writes_obs_column_and_copy_is_isolated(self):
        adata = _make_adata()
        out = _splitter(split_key="my_split").split(adata, copy=True)
        assert "my_split" in out.obs.columns
        assert "my_split" not in adata.obs.columns  # copy=True leaves the input's obs untouched
        assert out.X is adata.X  # ...and shares the cell matrix: obs-only copy, no data copied

    def test_split_refuses_to_overwrite_an_existing_split(self):
        """A split already in obs is someone's hold-out decision; replacing it silently is data loss."""
        adata = _splitter().split(_make_adata())
        with pytest.raises(ValueError, match="refusing to overwrite"):
            _splitter(seed=1).split(adata)

    def test_an_empty_hold_out_warns(self):
        """`floor(fraction * k)` rounds to 0 for every small stratum -- a split that silently didn't split."""
        adata = _make_adata()
        # A and B carry 6 perturbed drugs each; at 1/6 per stratum floor(1/6 * 6) == 1, so go below that.
        with pytest.warns(UserWarning, match="nothing was held out"):
            labels = _splitter(test_fraction=0.1).assign(adata)
        assert set(labels.unique()) == {"train", "control"}

    def test_always_train_covering_every_group_key_raises(self):
        """Each stratum would be one combination, so no test_fraction could ever hold anything out."""
        with pytest.raises(ValueError, match="covers every group key"):
            _splitter(always_train_keys=["cell_line", "drug"])
        # ...unless no hold-out was asked for in the first place.
        _splitter(always_train_keys=["cell_line", "drug"], test_fraction=0.0)

    def test_invalid_fraction_raises(self):
        with pytest.raises(ValueError, match="test_fraction"):
            _splitter(test_fraction=1.0)

    def test_always_train_keys_must_be_subset(self):
        """The message names the offending key and both parameters, not just that something is wrong."""
        with pytest.raises(ValueError, match=r"always_train_keys entries not found in group_keys: \['drug'\]"):
            CombinationSplitter(group_keys=["cell_line"], always_train_keys=["drug"])

    def test_missing_column_raises(self):
        with pytest.raises(KeyError, match="nonexistent"):
            CombinationSplitter(group_keys=["cell_line", "nonexistent"], always_train_keys=["cell_line"]).assign(
                _make_adata()
            )
