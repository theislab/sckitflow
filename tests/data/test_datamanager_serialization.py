"""A ``DataManager``'s group encoders serialize safely and reproducibly.

Encoders are frozen ``GroupEncoder`` (``Component``) dataclasses -- no callables -- so the manager
round-trips through both ``pickle`` and ``cloudpickle`` (the fitted transformer is rebuilt on demand via
``build``) AND exports a portable, pickle-free ``{type, version, config}`` spec. A pinned vocabulary
(``classes=`` / ``categories=``) makes the encoding reproducible regardless of the data ``build`` later
sees, so a reloaded manager cannot silently drift its categorical mapping at predict time.
"""

import json
import pickle

import cloudpickle
import numpy as np

from sc_flow.data._group_encoders import Affine, GroupEncoder, GroupEncoderContext, Label, OneHot
from sc_flow.data._manager import DataManager


def test_functional_group_encoder_round_trips():
    dm = DataManager(groups=("cell_line",), groups_encoding={"cell_line": Affine(scale=2.0, shift=1.0)})

    for restored in (pickle.loads(pickle.dumps(dm)), cloudpickle.loads(cloudpickle.dumps(dm))):
        enc = restored.groups_data_schema.groups_encoders["cell_line"]
        assert enc == Affine(scale=2.0, shift=1.0)
        fitted = enc.build(GroupEncoderContext(np.arange(3.0)))
        np.testing.assert_allclose(fitted.transform(np.arange(3.0)), np.arange(3.0) * 2.0 + 1.0)


def test_group_encoder_spec_round_trips():
    """Every group encoder exports a portable JSON ``{type, version, config}`` spec (no pickle needed)."""
    for enc in (Affine(scale=2.0, shift=1.0), Label(), Label(classes=("a", "b")), OneHot(categories=("x", "y", "z"))):
        spec = enc.to_spec()
        json.dumps(spec)  # portable: pure JSON, no callables / no pickled estimator
        assert GroupEncoder.from_spec(spec) == enc


def test_pinned_one_hot_mapping_is_data_independent():
    """A pinned vocabulary fixes the encoding, so a reloaded manager cannot drift it at predict time."""
    enc = OneHot(categories=("A", "B", "C"))
    probe = np.array(["C", "A"]).reshape(-1, 1)

    first = enc.build(GroupEncoderContext(np.array(["A", "B"])))
    second = enc.build(GroupEncoderContext(np.array(["B", "B", "C"])))  # different, reordered build data

    np.testing.assert_array_equal(list(first.categories_[0]), ["A", "B", "C"])
    np.testing.assert_allclose(first.transform(probe).toarray(), second.transform(probe).toarray())
