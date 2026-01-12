from collections.abc import Callable
from typing import Any, Literal

import torchsde
from torch import Tensor, device
from torchsde import sdeint

from sc_flow.backends.torch.solvers.solver import Solver


class SDESolver(Solver):
    r"""Class for solving stochastic differential equations (SDEs) with TorchSDE.

    :param sde_type: (Optional) Specifies whether the SDE should be interpreted as
        an Itô or Stratonovich SDE. Choices are ``"ito"`` or ``"stratonovich"``.
        Defaults to ``"ito"``.
    :type sde_type: class:`Literal["ito", "stratonovich"]`

    :param noise_type: (Optional) Specifies the structure of the diffusion noise.
        Choices are: ``"scalar"``, ``"diagonal"``, ``"general"``, or ``"additive"``.
        Defaults to ``"diagonal"``.
    :type noise_type: class:`Literal["scalar", "diagonal", "general", "additive"]`

    :param method: (Optional) Integration method used by TorchSDE. When ``None``
        it defaults to ``"euler"``.
    :type method: class:`str | None`

    :param device_id: (Optional) Identifier for the device on which the SDE will be solved. Choices are ``"cpu"`` or ``"cuda"``. Defaults to ``"cpu"``.
    :type device_id: class:`Literal["cuda", "cpu"]`
    """

    def __init__(
        self,
        sde_type: Literal["ito", "stratonovich"] = "ito",
        noise_type: Literal["scalar", "diagonal", "general", "additive"] = "diagonal",
        method: str | None = None,
        device_id: Literal["cuda", "cpu"] = "cpu",
    ):
        super().__init__()
        self.method = method or "euler"
        self.sde_type = torchsde.SDEIto if sde_type == "ito" else torchsde.SDEStratonovich
        self.noise_type = noise_type
        self.device = device(device_id)

    def solve(
        self,
        source,
        time,
        drift_fn: Callable[[Tensor, Tensor], Tensor],
        diffusion_fn: Callable[[Tensor, Tensor], Tensor],
        *,
        rtol: float,
        atol: float,
        return_trajectory: bool = True,
        solver_kwargs: dict[str, Any],
    ) -> Any:
        r"""Integrates the SDE specified by the drift and diffusion functions.

        :param source: Initial state for the SDE.
        :type source: class:`torch.Tensor`

        :param time: Time grid over which to integrate.
        :type time: class:`torch.Tensor`

        :param drift_fn: Callable implementing the drift term :math:`f(t, x)`.
        :type drift_fn: class:`Callable[[Tensor, Tensor], Tensor]`

        :param diffusion_fn: Callable implementing the diffusion term :math:`g(t, x)`.
        :type diffusion_fn: class:`Callable[[Tensor, Tensor], Tensor]`

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
        source = source.to(self.device)
        time = time.to(self.device)

        _noise_type = self.noise_type
        _SDE_base = self.sde_type

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
            source,
            time,
            rtol=rtol,
            atol=atol,
            method=self.method,
            **solver_kwargs,
        )
        if return_trajectory:
            return trajectory
        else:
            return trajectory[-1]
