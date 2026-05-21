import functools
import inspect
import warnings
from typing import Any

import diffrax as dfx
import jax
import jax.numpy as jnp
import lineax as lx
from diffrax import ControlTerm, ODETerm
from jaxtyping import PyTree

from sc_flow.backends.jax._types import ArrayLike, TDevice, TDiffusion, TSDEDynamics, TTimeStateDiffusion, TVfFn
from sc_flow.backends.jax._utils import get_sde_solver
from sc_flow.backends.jax.solvers._solver import BaseSolver


class SDESolver(BaseSolver[TSDEDynamics]):
    r"""Class for solving neural Stochastic Differential Equations (SDEs).

    :param dynamics: Encodes the drift (of type BaseVelocityField) and the diffusion terms of the SDE.

    :type dynamics: class:`TSDEDynamics`

    :param method: (Optional) The numerical integration scheme used for the SDE.
        When ``None`` it defaults to :class:`diffrax.Euler`.
    :type method: class: `diffrax.AbstractSolver | None`

    :param num_time_steps: Number of time steps used to construct the time grid over
        :math:`[0, 1]`. When :param:`dt0` is not explicitly passed via :param:`solver_kwargs`,
        the initial step size is set to ``1 / (num_time_steps - 1)``.
    :type num_time_steps: class: `int`

    :param device_id: Identifier for the JAX device on which the SDE is solved.
    :type device_id: class: ` TDevice`
    """

    def __init__(
        self,
        dynamics: TSDEDynamics,
        *,
        method: str | dfx.AbstractSolver | None = None,
        device_id: TDevice = "cpu",
        vf_kwargs: dict[str, Any] | None = None,
        df_kwargs: PyTree[Any] | None = None,
    ) -> None:
        super().__init__(dynamics=dynamics, method=method, device_id=device_id)
        self._method = get_sde_solver(method)
        vf_kwargs = vf_kwargs or {}
        self._drift_fn = dynamics[0].get_vf_fn(**vf_kwargs)
        self._diffusion_fn = self._diffusion_fn_wrapper(dynamics[1], df_kwargs)

    def solve(
        self,
        source: ArrayLike,
        t0: float = 0.0,
        t1: float = 1.0,
        brownian_motion: dfx.AbstractBrownianPath | None = None,
        bm_tol: float = 1e-3,
        *,
        num_time_steps: int = 500,
        return_trajectory: bool = False,
        solver_kwargs: dict[str, Any] | None = None,
    ) -> ArrayLike:
        r"""Solve the SDE defined by drift and diffusion vector fields.

        :param source: Initial state :math:`x_{t=0}` from which the SDE is integrated.
        :type source: class: `ArrayLike`

        :param t0: Initial time for the SDE integration. Defaults to ``0.0``.
        :type t0: class: `float`

        :param t1: Final time for the SDE integration. Defaults to ``1.0``.
        :type t1: class: `float`

        :param brownian_motion: (Optional) Brownian path driving the diffusion term.
            When ``None``, a :class:`diffrax.VirtualBrownianTree` is constructed over
            :math:`[0, 1]` with shape ``source.shape`` and a default tolerance of ``1e-3``.
        :type brownian_motion: class: `diffrax.AbstractBrownianPath | None`

        :param bm_tol: Tolerance used for brownian motion when brownian_motion is ``None``.
            Defaults to ``1e-3``.
        :type bm_tol: class: `float`

        :param return_trajectory: When ``True``, the full solution trajectory is returned
            otherwise only the final state at :math:`t = 1`
            is returned.
        :type return_trajectory: class: `bool`

        :param solver_kwargs: (Optional) Dictionary of keyword arguments forwarded to
            :func:`diffrax.diffeqsolve`. The following keys are handled explicitly and
            removed from this dictionary:

            * ``"dt0"``: Initial time step size. When not provided, it is set to
              ``1 / (num_time_steps - 1)``.
            * ``"max_steps"``: Maximum number of internal solver steps. Defaults to
              ``10_000``.
            * ``"stepsize_controller"``: Stepsize controller used by the solver.
              Defaults to :class:`diffrax.ConstantStepSize`. This must be compatible
              with the chosen Brownian path and adjoint settings; otherwise
              :func:`validate_sde_solve_params` will raise an error.
            * ``"adjoint"``: (Optional) Adjoint method configuration passed on to
              :func:`diffrax.diffeqsolve` (e.g. for backpropagation through the SDE).

            Any remaining entries in :param:`solver_kwargs` are passed to
            :func:`diffrax.diffeqsolve`.
        :type solver_kwargs: class: `dict[str, Any] | None`

        :returns: Either the full solution trajectory over :attr:`ts` or the final state at
            :math:`t = 1`, depending on :param:`return_trajectory`.
        :rtype: class: `ArrayLike`
        """
        config = self._prepare_solve_config(
            source=source,
            t0=t0,
            t1=t1,
            num_time_steps=num_time_steps,
            return_trajectory=return_trajectory,
            solver_kwargs=solver_kwargs,
        )

        adjoint = config.remaining_kwargs.pop("adjoint", None)

        if brownian_motion is None:
            brownian_motion = dfx.VirtualBrownianTree(
                t0=t0,
                t1=t1,
                shape=config.source_on_device.shape,
                tol=bm_tol,
                key=jax.random.PRNGKey(0),
            )

        self._validate_sde_config(
            brownian_motion=brownian_motion,
            adjoint=adjoint,
            stepsize_controller=config.stepsize_controller,
        )

        if adjoint is not None:
            config.remaining_kwargs["adjoint"] = adjoint

        terms = dfx.MultiTerm(
            ODETerm(self._drift_fn),
            ControlTerm(self._diffusion_fn, brownian_motion),
        )
        print(terms)
        print(config.source_on_device.shape)
        trajectory = dfx.diffeqsolve(
            terms,
            solver=self._method,
            t0=t0,
            t1=t1,
            dt0=config.dt0,
            y0=config.source_on_device,
            saveat=config.saveat,
            max_steps=config.max_steps,
            stepsize_controller=config.stepsize_controller,
            **config.remaining_kwargs,
        )
        if return_trajectory:
            return trajectory.ys
        else:
            return trajectory.ys[-1]

    def _diffusion_fn_wrapper(
        self,
        diffusion_fn: TDiffusion,
        df_kwargs: PyTree[Any] | None = None,
    ) -> TTimeStateDiffusion:
        """Wraps the diffusion function to ensure it has the correct signature for TorchSDE."""
        df_kwargs = df_kwargs or {}
        partial_diffusion = functools.partial(diffusion_fn, **df_kwargs)

        sig = inspect.signature(diffusion_fn)
        params = sig.parameters

        remaining_params = [p for p in params if p not in df_kwargs]
        num_params = len(remaining_params)

        def wrapped_diffusion(t: ArrayLike, y: ArrayLike, args: Any = None) -> ArrayLike:
            if num_params >= 2:
                out = partial_diffusion(t, y)
            else:
                out = partial_diffusion(t)

            if isinstance(out, lx.AbstractLinearOperator):
                return out

            if jnp.ndim(out) == 0:
                return lx.DiagonalLinearOperator(jnp.full_like(y, out))
            if out.shape == y.shape:
                return lx.DiagonalLinearOperator(out)
            if jnp.ndim(out) == 2:
                return lx.MatrixLinearOperator(out)

            return lx.PyTreeLinearOperator(out, jax.eval_shape(lambda: y))

        return wrapped_diffusion

    def _validate_sde_config(
        self,
        brownian_motion: dfx.AbstractBrownianPath,
        adjoint: Any,
        stepsize_controller: dfx.AbstractStepSizeController,
    ) -> None:
        """Validate SDE-specific configuration parameters."""
        if isinstance(brownian_motion, dfx.UnsafeBrownianPath):
            if adjoint is None:
                msg = "Cannot use UnsafeBrownianPath with adjoint=None.\n Default adjoint = RecursiveCheckpointAdjoint is not compatible with UnsafeBrownianPath. Use a different adjoint method, such as ForwardMode()"
                raise ValueError(msg)
            elif not isinstance(stepsize_controller, dfx.AbstractAdaptiveStepSizeController):
                msg = (
                    "Using UnsafeBrownianPath with fixed-step solver:\n - Backpropagation through the SDE will not be supported.\n- Output is nondeterministic w.r.t RNG key due to floating-point fluctuations.\n",
                )
                warnings.warn(msg, stacklevel=2)

    @property
    def drift_fn(self) -> TVfFn:
        """Returns the drift function used in the SDE."""
        return self._drift_fn

    @property
    def diffusion_fn(self) -> TTimeStateDiffusion:
        """Returns the diffusion function used in the SDE."""
        return self._diffusion_fn
