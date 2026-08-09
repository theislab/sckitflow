"""Unit tests for :func:`sckitflow.core._data_utils.subscript_step_data`.

The contract under test: a matching permutation reindexes each side of a batch *independently*, and
every per-observation field of a side moves with it -- including ``target_response_data``, which the
model never consumes but which must stay row-aligned with ``target_state`` for output reconstruction.
"""

import torch

from sckitflow.core._data_utils import subscript_step_data
from sckitflow.core._types import new_step_data


def _batch(n: int = 4) -> dict:
    """A StepData whose every field encodes its row index, so a permutation is visible in the values."""
    rows = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    return new_step_data(
        target_state=rows.clone(),
        target_coupling_lin=rows.clone() + 100,
        target_condition_data={"drug": rows.clone() + 200, "absent": None},
        target_group_data={"line": rows.clone() + 300},
        target_response_data={"ytgt": rows.clone() + 400},
        source_state=rows.clone() + 1000,
        source_coupling_lin=rows.clone() + 1100,
        source_condition_data={"drug": rows.clone() + 1200},
    )


def _col(field) -> list[float]:
    return field.squeeze(1).tolist()


class TestSubscriptStepData:
    def test_each_side_is_indexed_independently(self):
        out = subscript_step_data(_batch(), src_idxs=torch.tensor([3, 2]), tgt_idxs=torch.tensor([0, 1]))

        assert _col(out["target_state"]) == [0.0, 1.0]
        assert _col(out["target_coupling_lin"]) == [100.0, 101.0]
        assert _col(out["source_state"]) == [1003.0, 1002.0]
        assert _col(out["source_coupling_lin"]) == [1103.0, 1102.0]

    def test_dict_fields_are_indexed_value_by_value(self):
        out = subscript_step_data(_batch(), src_idxs=torch.tensor([1]), tgt_idxs=torch.tensor([2]))

        assert _col(out["target_condition_data"]["drug"]) == [202.0]
        assert _col(out["target_group_data"]["line"]) == [302.0]
        assert _col(out["source_condition_data"]["drug"]) == [1201.0]
        assert out["target_condition_data"]["absent"] is None  # a None value survives indexing

    def test_response_data_stays_aligned_with_the_target_state(self):
        """The regression this field exists for: covariates carried for output must follow the permutation."""
        perm = torch.tensor([3, 0, 2, 1])
        out = subscript_step_data(_batch(), tgt_idxs=perm)

        states = _col(out["target_state"])
        responses = _col(out["target_response_data"]["ytgt"])
        assert responses == [state + 400 for state in states]

    def test_a_none_index_leaves_that_side_untouched(self):
        batch = _batch()
        out = subscript_step_data(batch, src_idxs=None, tgt_idxs=torch.tensor([1, 0]))

        assert out["source_state"] is batch["source_state"]  # same object, not merely equal
        assert _col(out["target_state"]) == [1.0, 0.0]

    def test_none_fields_stay_none_and_untouched_keys_survive(self):
        batch = _batch()
        out = subscript_step_data(batch, src_idxs=torch.tensor([0]), tgt_idxs=torch.tensor([0]))

        assert out["source_group_data"] is None  # never set, never indexed
        assert out["target_coupling_quad"] is None
        assert set(out) == set(batch)  # the returned batch is still a complete StepData

    def test_the_input_batch_is_not_mutated(self):
        batch = _batch()
        subscript_step_data(batch, src_idxs=torch.tensor([2]), tgt_idxs=torch.tensor([2]))

        assert _col(batch["target_state"]) == [0.0, 1.0, 2.0, 3.0]
        assert _col(batch["target_response_data"]["ytgt"]) == [400.0, 401.0, 402.0, 403.0]
