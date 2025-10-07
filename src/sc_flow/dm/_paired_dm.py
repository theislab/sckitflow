from typing import NewType

import numpy as np

from sc_flow._types import MappedArray
from sc_flow.dm._unconditional_dm import UnconditionalDataManager

## TODO: remove when felix writes this stuff
StateData = NewType("StateData", np.array)
ConditionData = NewType("ConditionData", MappedArray)
TargetData = NewType("TargetData", MappedArray)
CouplingData = NewType("CouplingData", MappedArray)


class PairedConditionalDataManager(UnconditionalDataManager):
    """"""  # noqa
