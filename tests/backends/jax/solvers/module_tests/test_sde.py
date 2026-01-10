import warnings

import diffrax as dfx
import jax.numpy as jnp
import pytest
from pytest import MonkeyPatch as mp

import sc_flow.backends.jax.solvers.sde_solver as sde_solver_module
from sc_flow.backends.jax.solvers.sde_solver import SDESolver


def _test_drift(t, y):
    return jnp.zeros_like(y)


def _test_diffusion(t, y):
    return jnp.ones_like(y)


def _test_patch_diffeqsolve(monkeypatch: mp) -> None:
    class _test_MultiTerm:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class _test_Term:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    def _test_diffeqsolve(*args, **kwargs):
        class _Result:
            ys = jnp.zeros((2,))

        return _Result()

    monkeypatch.setattr(sde_solver_module.dfx, "MultiTerm", _test_MultiTerm)
    monkeypatch.setattr(sde_solver_module.dfx, "diffeqsolve", _test_diffeqsolve)
    monkeypatch.setattr(sde_solver_module, "ODETerm", _test_Term)
    monkeypatch.setattr(sde_solver_module, "ControlTerm", _test_Term)


def test_sde_solver_unsafebrownian_with_adaptive_controller_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _test_AbstractStepSizeController:
        pass

    class _test_AdaptiveStepSizeController(_test_AbstractStepSizeController):
        pass

    class _test_UnsafeBrownian:
        pass

    monkeypatch.setattr(
        sde_solver_module.dfx,
        "AbstractStepSizeController",
        _test_AbstractStepSizeController,
    )
    monkeypatch.setattr(
        sde_solver_module.dfx,
        "AbstractAdaptiveStepSizeController",
        _test_AdaptiveStepSizeController,
    )
    monkeypatch.setattr(
        sde_solver_module.dfx,
        "UnsafeBrownianPath",
        _test_UnsafeBrownian,
    )

    _test_patch_diffeqsolve(monkeypatch)

    solver = SDESolver(method=dfx.Euler(), num_time_steps=10, device_id="cpu")
    source = jnp.zeros((2,))
    brownian = _test_UnsafeBrownian()
    controller = _test_AdaptiveStepSizeController()

    with pytest.raises(ValueError, match="Cannot use an UnsafeBrownianPath"):
        solver.solve(
            source=source,
            drift_fn=_test_drift,
            diffusion_fn=_test_diffusion,
            brownian_motion=brownian,
            return_trajectory=False,
            solver_kwargs={"stepsize_controller": controller},
        )


def test_sde_solver_unsafebrownian_with_fixed_stepsize_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _test_AbstractStepSizeController:
        pass

    class _test_AdaptiveStepSizeController(_test_AbstractStepSizeController):
        pass

    class _test_FixedStepSizeController(_test_AbstractStepSizeController):
        pass

    class _test_UnsafeBrownian:
        pass

    monkeypatch.setattr(
        sde_solver_module.dfx,
        "AbstractStepSizeController",
        _test_AbstractStepSizeController,
    )
    monkeypatch.setattr(
        sde_solver_module.dfx,
        "AbstractAdaptiveStepSizeController",
        _test_AdaptiveStepSizeController,
    )
    monkeypatch.setattr(
        sde_solver_module.dfx,
        "UnsafeBrownianPath",
        _test_UnsafeBrownian,
    )

    _test_patch_diffeqsolve(monkeypatch)

    solver = SDESolver(method=dfx.Euler(), num_time_steps=10, device_id="cpu")
    source = jnp.zeros((2,))
    brownian = _test_UnsafeBrownian()
    controller = _test_FixedStepSizeController()

    dummy_adjoint = object()

    with pytest.warns(
        UserWarning,
        match="Using UnsafeBrownianPath with fixed-step solver",
    ):
        _ = solver.solve(
            source=source,
            drift_fn=_test_drift,
            diffusion_fn=_test_diffusion,
            brownian_motion=brownian,
            return_trajectory=False,
            solver_kwargs={
                "stepsize_controller": controller,
                "adjoint": dummy_adjoint,
            },
        )


def test_sde_solver_safe_brownian_does_not_warn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _test_AbstractStepSizeController:
        pass

    class _test_AdaptiveStepSizeController(_test_AbstractStepSizeController):
        pass

    class _test_SafeBrownian:
        pass

    class _test_UnsafeBrownian:
        pass

    monkeypatch.setattr(
        sde_solver_module.dfx,
        "AbstractStepSizeController",
        _test_AbstractStepSizeController,
    )
    monkeypatch.setattr(
        sde_solver_module.dfx,
        "AbstractAdaptiveStepSizeController",
        _test_AdaptiveStepSizeController,
    )
    monkeypatch.setattr(
        sde_solver_module.dfx,
        "UnsafeBrownianPath",
        _test_UnsafeBrownian,
    )

    _test_patch_diffeqsolve(monkeypatch)

    solver = SDESolver(method=dfx.Euler(), num_time_steps=10, device_id="cpu")
    source = jnp.zeros((2,))
    brownian = _test_SafeBrownian()
    controller = _test_AdaptiveStepSizeController()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = solver.solve(
            source=source,
            drift_fn=_test_drift,
            diffusion_fn=_test_diffusion,
            brownian_motion=brownian,
            return_trajectory=False,
            solver_kwargs={"stepsize_controller": controller},
        )

    assert len(w) == 0, "Safe Brownian + any stepsize controller must not emit warnings."
