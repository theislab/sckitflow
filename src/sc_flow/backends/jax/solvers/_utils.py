import warnings
from typing import Any

import diffrax as dfx


def validate_sde_solve_params(stepsize_controller: Any, brownian_motion: Any, adjoint: Any) -> None:
    """Validate the parameters for solving SDEs."""
    if stepsize_controller is not None:
        if not isinstance(stepsize_controller, dfx.AbstractStepSizeController):
            raise TypeError("options['stepsize_controller'] must be an instance of diffrax.AbstractStepSizeController.")
    if isinstance(brownian_motion, dfx.UnsafeBrownianPath) and isinstance(
        stepsize_controller, dfx.AbstractAdaptiveStepSizeController
    ):
        raise ValueError("Cannot use an UnsafeBrownianPath with an adaptive stepsize controller.")

    elif isinstance(brownian_motion, dfx.UnsafeBrownianPath) and not isinstance(
        stepsize_controller, dfx.AbstractAdaptiveStepSizeController
    ):
        warnings.warn(
            "Using UnsafeBrownianPath with fixed-step solver:\n"
            "- Backpropagation through the SDE will not be supported.\n"
            "- Output is nondeterministic w.r.t RNG key due to floating-point fluctuations.\n",
            stacklevel=2,
        )

    if isinstance(brownian_motion, dfx.UnsafeBrownianPath) and adjoint is None:
        raise ValueError(
            "Default adjoint method is RecursiveCheckPointAdjoint, which is incompatible with UnsafeBrownianPath. Please specify appropiate adjoint methods such as ForwardMode() in the solver_kwargs"
        )
