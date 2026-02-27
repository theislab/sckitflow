import functools
import inspect
from typing import Any

import torchsde
from torch import Tensor
from torchsde import sdeint

from sc_flow.backends.torch._types import TDevice, TDiffusion, TNoiseType, TSDEDynamics, TSDEType, TVfFn
from sc_flow.backends.torch.solvers.solver import BaseSolver


class SDESolver(BaseSolver[TSDEDynamics]):
    r"""Class for solving stochastic differential equations (SDEs) with TorchSDE.

    :param dynamics: Encodes the drift (of type BaseVelocityField) and the diffusion terms of the SDE
    :type dynamics: class:`TSDEDynamics`

    :param sde_type: (Optional) Specifies whether the SDE should be interpreted as
        an Itô or Stratonovich SDE. Choices are ``"ito"`` or ``"stratonovich"``.
        Defaults to ``"ito"``.
    :type sde_type: class:`Literal["ito", "stratonovich"]`

    :param noise_type: (Optional) Specifies the structure of the diffusion noise.
        Choices are: ``"scalar"``, ``"diagonal"``, ``"general"``, or ``"additive"``.
        Defaults to ``"diagonal"``.
    :type noise_type: class:`TNoiseType`

    :param method: (Optional) Integration method used by TorchSDE. When ``None``
        it defaults to ``"euler"``.
    :type method: class:`TSDEType`

    :param device_id: (Optional) Identifier for the device on which the SDE will be solved.
    :type device_id: class:`TDevice`

    :param vf_kwargs: (Optional) Keyword arguments passed to
            :meth:`BaseVelocityField.get_vf_fn`.
    :type vf_kwargs: class:`dict[str, Any] | None`
    """

    def __init__(
        self,
        dynamics: TSDEDynamics,
        *,
        sde_type: TSDEType = "ito",
        noise_type: TNoiseType = "diagonal",
        method: str | None = None,
        device_id: TDevice = "cpu",
        vf_kwargs: dict[str, Any] | None = None,
        df_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(dynamics=dynamics, method=method, device_id=device_id)

        self._sde_type = torchsde.SDEIto if sde_type == "ito" else torchsde.SDEStratonovich
        self._noise_type = noise_type

        vf_kwargs = vf_kwargs or {}
        self._drift_fn = dynamics[0].get_vf_fn(**vf_kwargs)
        self._diffusion_fn = self._get_diffusion_fn_wrapper(dynamics[1], df_kwargs)

    def solve(
        self,
        source,
        time,
        *,
        rtol: float,
        atol: float,
        return_trajectory: bool = False,
        solver_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        r"""Integrates the SDE specified by the drift and diffusion functions.

        :param source: Initial state for the SDE.
        :type source: class:`torch.Tensor`

        :param time: Time grid over which to integrate.
        :type time: class:`torch.Tensor`

        :param rtol: Relative tolerance used by TorchSDE.
        :type rtol: class:`float`

        :param atol: Absolute tolerance used by TorchSDE.
        :type atol: class:`float`

        :param return_trajectory: When ``True``, returns the full trajectory, else returns only the final state at the last time point.
        :type return_trajectory: class:`bool`

        :param solver_kwargs: Additional keyword arguments forwarded to
            :func:`torchsde.sdeint`.
        :type solver_kwargs: class:`dict[str, Any]`

        :returns: Output of :func:`torchsde.sdeint`, typically either the full state
            trajectory or the final integration point.
        :rtype: class:`Any`
        """
        config = self._prepare_solve_config(source, time, solver_kwargs)

        _noise_type = self._noise_type
        _SDE_base = self._sde_type

        drift_fn = self._drift_fn
        diffusion_fn = self._diffusion_fn

        class _BaseSDE(_SDE_base):
            """SDE class for torchsde integration."""

            def __init__(self):
                super().__init__(noise_type=_noise_type)

            def f(self, t: Tensor, y: Tensor) -> Tensor:
                return drift_fn(t, y)

            def g(self, t: Tensor, y: Tensor) -> Tensor:
                return diffusion_fn(t, y)

        sde = _BaseSDE()

        trajectory = sdeint(
            sde,
            config.source_on_device,
            config.time_on_device,
            rtol=rtol,
            atol=atol,
            method=self._method,
            **config.remaining_kwargs,
        )
        if return_trajectory:
            return trajectory
        else:
            return trajectory[-1]

    @property
    def sde_type(self) -> TSDEType:
        """Returns the SDE type (Itô or Stratonovich)."""
        return self._sde_type

    @property
    def noise_type(self) -> TNoiseType:
        """Returns the noise type used in the SDE."""
        return self._noise_type

    @property
    def drift_fn(self) -> TVfFn:
        """Returns the drift function used in the SDE."""
        return self._drift_fn

    @property
    def diffusion_fn(self) -> TDiffusion:
        """Returns the diffusion function used in the SDE."""
        return self._diffusion_fn

    def _get_diffusion_fn_wrapper(
        self,
        diffusion_fn: TDiffusion,
        df_kwargs: dict[str, Any] | None = None,
    ) -> TVfFn:
        """Wraps the diffusion function to ensure it has the correct signature for TorchSDE."""
        df_kwargs = df_kwargs or {}
        partial_diffusion = functools.partial(diffusion_fn, **df_kwargs)
        sig = inspect.signature(diffusion_fn)
        params = sig.parameters
        remaining_params = [p for p in params.values() if p.name not in df_kwargs]
        num_params = len(remaining_params)

        def wrapped_diffusion(t: Tensor, y: Tensor, args: Any = None) -> Tensor:
            if num_params >= 2:
                return partial_diffusion(t, y)
            else:
                return partial_diffusion(t)

        return wrapped_diffusion
