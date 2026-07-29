"""A ``DataManager`` carrying a functional group encoder is serializable.

The encoder is a frozen dataclass ``GroupEncoder`` (no callables), so the manager --
and any ``SCFlow`` model that embeds it -- round-trips through both ``pickle`` and
``cloudpickle``, and the fitted transformer is rebuilt on demand via ``build``.
"""

import pickle

import cloudpickle
import numpy as np

from sc_flow.data._group_encoders import Affine, GroupEncoderContext
from sc_flow.data._manager import DataManager


def test_functional_group_encoder_round_trips():
    dm = DataManager(groups=("cell_line",), groups_encoding={"cell_line": Affine(scale=2.0, shift=1.0)})

    for restored in (pickle.loads(pickle.dumps(dm)), cloudpickle.loads(cloudpickle.dumps(dm))):
        enc = restored.groups_data_schema.groups_encoders["cell_line"]
        assert enc == Affine(scale=2.0, shift=1.0)
        fitted = enc.build(GroupEncoderContext(np.arange(3.0)))
        np.testing.assert_allclose(fitted.transform(np.arange(3.0)), np.arange(3.0) * 2.0 + 1.0)
