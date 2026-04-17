# TODO: remove after merging with the solver branch
from typing import Any, Literal

import numpy as np
import pytest
import torch
from torch import Tensor, device
from torchdiffeq import odeint

from sc_flow.backends.torch.methods import BaseMethod
from sc_flow.backends.torch.nn._vf import BaseVelocityField, MLPVelocity
from sc_flow.backends.torch.probability_paths import LinearDiracProbabilityPath


class ODESolver(torch.nn.Module):
    r"""Class for solving deterministic ordinary differential equations (ODEs) with :func:`torchdiffeq.odeint`.

    :param method: (Optional) Integration scheme used by ``torchdiffeq``. Defaults to ``"euler"``. Other valid options depend on ``torchdiffeq``.
    :type method: class:`str`

    :param device_id: (Optional) Identifier for the target compute device. Choices are ``"cpu"`` or ``"cuda"``. Defaults to ``"cpu"``.
    :type device_id: class:`Literal["cuda", "cpu"]`
    """

    def __init__(
        self,
        method: str = "euler",
        device_id: Literal["cuda", "cpu"] = "cpu",
    ) -> None:
        super().__init__()
        self.method = method
        self.device = device(device_id)

    def solve(
        self,
        vf: BaseVelocityField,
        source: Tensor,
        time: Tensor,
        *,
        rtol: float,
        atol: float,
        vf_kwargs: dict[str, Any] | None = None,
        solver_kwargs: dict[str, Any] | None = None,
        return_trajectory: bool = True,
    ) -> Tensor:
        r"""Integrates the ODE defined by the provided velocity field.

        :param vf: Velocity field providing the time-dependent dynamics. Must implement :meth:`BaseVelocityField.get_vf_fn`.
        :type vf: class:`BaseVelocityField`

        :param source: Initial state of the ODE.
        :type source: class:`torch.Tensor`

        :param time: Time grid over which to integrate the ODE.
        :type time: class:`torch.Tensor`

        :param rtol: Relative tolerance for the ODE solver.
        :type rtol: class:`float`

        :param atol: Absolute tolerance for the ODE solver.
        :type atol: class:`float`

        :param vf_kwargs: (Optional) Keyword arguments passed to
            :meth:`BaseVelocityField.get_vf_fn`.
        :type vf_kwargs: class:`dict[str, Any] | None`

        :param solver_kwargs: (Optional) Keyword arguments forwarded directly to
            :func:`torchdiffeq.odeint`.
        :type solver_kwargs: class:`dict[str, Any] | None`

        :param return_trajectory: When ``True``, returns the full trajectory, else returns only the final state at the last time point.
        :type return_trajectory: class:`bool`

        :returns: Either the full state trajectory or the final state depending on
            :param:`return_trajectory`.
        :rtype: class:`torch.Tensor`
        """
        if solver_kwargs is None:
            solver_kwargs = {}

        if vf_kwargs is None:
            vf_kwargs = {}

        diff_eqn = vf.get_vf_fn(**vf_kwargs)

        source = source.to(self.device)
        time = time.to(self.device)

        trajectory = odeint(
            diff_eqn,
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


# TODO replace with actual independent coupling function once merged
def independent_coupling(
    source: torch.Tensor,
    target: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    src_random_perm_idx = np.random.choice(np.arange(source.shape[0]), size=source.shape[0], replace=False)
    tgt_random_perm_idx = np.random.choice(np.arange(target.shape[0]), size=source.shape[0], replace=False)
    min_shape = min(src_random_perm_idx.shape[0], tgt_random_perm_idx.shape[0])
    return src_random_perm_idx[:min_shape], tgt_random_perm_idx[:min_shape]


batch_size = 4
dim = 3


def dummy_time_sampler(n):
    return torch.rand(size=(n,))


class TestTorchMethods:
    @pytest.mark.parametrize("generate_from_noise", [True, False])
    @pytest.mark.parametrize("control_key", [None, "src_cell_data"])
    def test_step_fn_runs_without_error(self, generate_from_noise, control_key):
        vf = MLPVelocity(
            state_dim=3,
            state_encoder_output_dim=32,
            encode_state=True,
            encode_time=True,
        )
        probability_path = LinearDiracProbabilityPath(sigma=1.0)
        match_fn = independent_coupling
        time_sampler = dummy_time_sampler
        target_dim = 3
        opt = torch.optim.AdamW
        method = BaseMethod(
            vf=vf,
            probability_path=probability_path,
            time_sampler=time_sampler,
            solver=ODESolver(method="euler"),
            match_fn=match_fn,
            target_dim=target_dim,
            generate_from_noise=generate_from_noise,
            control_key=control_key,
            optimizer=opt,
            condition_dict={"drug": torch.ones((batch_size, 1))},
        )

        # Create dummy batch
        batch = {
            "src_cell_data": torch.ones((batch_size, dim)),
            "tgt_cell_data": torch.ones((batch_size, dim)) * 2,
            "condition": {"dummy_cond": torch.ones((batch_size, 1))},
        }
        loss = method.step_fn(batch)

        assert (loss.ndim == 0) or np.isscalar(loss)
        assert torch.isfinite(loss)

    @pytest.mark.parametrize("generate_from_noise", [True, False])
    @pytest.mark.parametrize("batched", [True])
    def test_predict_batched(self, generate_from_noise, batched):
        state_dim = 3
        vf = MLPVelocity(
            state_dim=state_dim,
        )
        probability_path = LinearDiracProbabilityPath(sigma=1.0)
        match_fn = independent_coupling
        time_sampler = dummy_time_sampler
        target_dim = dim
        opt = torch.optim.AdamW

        method = BaseMethod(
            vf=vf,
            probability_path=probability_path,
            time_sampler=time_sampler,
            match_fn=match_fn,
            target_dim=target_dim,
            solver=ODESolver(method="euler"),
            generate_from_noise=generate_from_noise,
            control_key=None,
            optimizer=opt,
            condition_dict={"drug": torch.ones((batch_size, 1))},
        )
        src = {
            "drug_1": torch.randn(size=(batch_size, dim)),
            "drug_2": torch.randn(size=(batch_size, dim)),
        }
        cond = {
            "drug_1": {"drug": torch.randn((1, 1, 3))},
            "drug_2": {"drug": torch.randn((1, 1, 3))},
        }

        x_pred = method.predict(x=src, condition=cond, batched=batched)

        assert x_pred["drug_1"].shape == (batch_size, state_dim)

    @pytest.mark.parametrize("generate_from_noise", [True, False])
    @pytest.mark.parametrize("batched", [False])
    def test_predict_not_batched(self, generate_from_noise, batched):
        state_dim = 3
        vf = MLPVelocity(
            state_dim=state_dim,
        )
        probability_path = LinearDiracProbabilityPath(sigma=1.0)
        match_fn = independent_coupling
        time_sampler = dummy_time_sampler
        target_dim = dim
        opt = torch.optim.AdamW

        method = BaseMethod(
            vf=vf,
            probability_path=probability_path,
            time_sampler=time_sampler,
            match_fn=match_fn,
            target_dim=target_dim,
            solver=ODESolver(method="euler"),
            generate_from_noise=generate_from_noise,
            control_key=None,
            optimizer=opt,
            condition_dict={"drug": torch.ones((batch_size, 1))},
        )
        src = torch.randn(size=(batch_size, dim))

        cond = {"drug": torch.randn((1, 1, 3))}

        x_pred = method.predict(x=src, condition=cond, batched=batched)
        assert x_pred.shape == (batch_size, state_dim)
