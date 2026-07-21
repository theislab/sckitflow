"""sc_flow.flow — the flow-matching toolbox (extends the ML core :mod:`sc_flow.core`).

Holds the flow-matching specifics: velocity fields, probability paths, the concrete loss objectives, the
ODE predict path, and the JAX/OTT optimal-transport coupling bridge. torch + Lightning are always the
backbone (in :mod:`sc_flow.core`); JAX is optional and pulled **lazily**, only when an OT coupling runs —
so importing this package (and the ``match_method="independent"`` path) needs no jax. Importing here
registers the ``fm-linear`` / ``otfm`` / ``genot`` objectives with the core registry.
"""

from sc_flow.core.training._objective import build_objective

# importing the concrete objectives registers fm-linear / otfm / genot with the core registry
from sc_flow.flow._objectives import GENOTObjective, LinearFMObjective, OTFMObjective
from sc_flow.flow._vf import BaseVelocityField, MLPVelocity

__all__ = [
    "build_objective",
    "LinearFMObjective",
    "OTFMObjective",
    "GENOTObjective",
    "BaseVelocityField",
    "MLPVelocity",
]
