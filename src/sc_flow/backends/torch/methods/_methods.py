from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

import sc_flow.methods as basemethods
from sc_flow.backends.torch._types import TensorLike
from sc_flow.backends.torch.nn._vf import BaseVelocityField
from sc_flow.backends.torch.probability_paths import BaseProbabilityPath

__all__ = ["BaseMethod"]


class Solver(ABC, nn.Module):
    """Abstract Base Class for Solvers"""

    @abstractmethod
    def solve(self, vf, source: torch.Tensor | None = None, **kwargs: Any) -> torch.Tensor:
        """Solve method to be implemented by subclasses."""
        pass


LinTerm = tuple[torch.Tensor, torch.Tensor]
QuadTerm = tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]
GENOTDataMatchFn = Callable[[LinTerm], torch.Tensor] | Callable[[QuadTerm], torch.Tensor]


class BaseMethod(basemethods.FlowMatching):
    """TODO."""

    def __init__(
        self,
        vf: BaseVelocityField,
        probability_path: BaseProbabilityPath,
        time_sampler: Callable[[int], torch.Tensor],  # [[int], np.ndarray],
        solver: Solver | Any,
        match_fn: Any,  # TODO: adapt type
        ema: int = 1,
        generate_from_noise: bool = False,
        control_key: str | None = None,
        noise_distribution: Callable[[tuple[int, ...]], torch.Tensor] | None = None,
        **kwargs: Any,  # TODO Add parcing of kwargs
    ):
        self.vf = vf
        self.probability_path = probability_path
        self.time_sampler = time_sampler
        self.match_fn = match_fn
        self.ema = ema
        self.generate_from_noise = generate_from_noise
        self.control_key = control_key
        self.solver = solver
        self.noise_distribution = noise_distribution or (lambda shape: torch.randn(size=shape))
        # TODO: add cfg

        self._is_trained = False
        self.vf_step_fn = self._get_vf_step_fn()

    @staticmethod
    def _prepare_data(
        batch: dict[str, Any],  # TODO adapt type
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor | None, ...],
    ]:
        """TODO"""
        src_lin, src_quad = batch.get("src_cell_data"), batch.get("src_cell_data_quad")
        tgt_lin, tgt_quad = batch.get("tgt_cell_data"), batch.get("tgt_cell_data_quad")

        if src_quad is None and tgt_quad is None:  # lin
            source, target = src_lin, tgt_lin
            arrs = src_lin, tgt_lin
        elif src_lin is None and tgt_lin is None:  # quad
            source, target = src_quad, tgt_quad
            arrs = src_quad, tgt_quad
        elif all(arr is not None for arr in (src_lin, tgt_lin, src_quad, tgt_quad)):  # fused quad
            source = torch.concat([src_lin, src_quad], dim=1)
            target = torch.concat([tgt_lin, tgt_quad], dim=1)
            arrs = src_quad, tgt_quad, src_lin, tgt_lin
        else:
            raise RuntimeError("Cannot infer OT problem type from data.")

        return (source, target), arrs  # type: ignore[return-value]

    def step_fn(
        self,
        batch: dict[str, Any],  # TODO Adapt types
    ) -> torch.Tensor:
        """Single step function of the solver.

        :param batch: Data batch with keys ``src_cell_data``, ``tgt_cell_data``, and
            ``condition``.
        :type batch: dict[str, torch.Tensor]
        """
        condition = batch.get("condition")
        (source, target), matching_data = self._prepare_data(batch)
        n = source.shape[0]
        time = self.time_sampler(n)

        src_ixs, tgt_ixs = self.match_fn(*matching_data)

        source, target = source[src_ixs], target[tgt_ixs]
        loss = self.vf_step_fn(
            time,
            source,
            target,
            condition,
        )
        return loss

    def _get_vf_step_fn(self):
        def vf_step_fn(
            time: torch.Tensor,
            source: torch.Tensor,
            target: torch.Tensor,
            conditions: dict[str, torch.Tensor] | Any,
        ):
            def loss_fn(
                t: torch.Tensor,
                source: torch.Tensor,
                target: torch.Tensor,
                conditions: dict[str, torch.Tensor],
            ) -> torch.Tensor:
                if self.generate_from_noise:
                    latent = self.noise_distribution(target.shape)
                    x_t = self.probability_path.compute_xt(t, latent, target)
                    if self.control_key is not None:
                        conditions = conditions.copy()
                        conditions[self.control_key] = source
                else:
                    x_t = self.probability_path.compute_xt(t, source, target)
                v_t = self.vf(
                    t,
                    x_t,
                    conditions,
                    source,
                )
                u_t = self.probability_path.compute_ut(t, x_t, source, target)
                return torch.mean((v_t - u_t) ** 2)

            with torch.autocast("cuda"):
                loss = loss_fn(time, source, target, conditions)
            loss.backward()
            return loss

        return vf_step_fn

    def train_step(
        self,
        batch: dict[str, TensorLike],
    ) -> float:
        """Training step of the method.

        :param batch: Data batch with keys ``src_cell_data``, ``tgt_cell_data``, and
            ``condition``.
        :type batch: dict[str, TensorLike]

        :param rng_step_fn: Random number generator for the step function. Default is None.
        :type rng_step_fn: TensorLike | None, optional
        """
        loss = self.step_fn(batch)
        return float(loss.cpu().numpy())

    def get_condition_embedding(self, condition: dict[str, torch.Tensor], return_as_numpy=True) -> torch.Tensor:
        pass

    def predict(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        condition: dict[str, torch.Tensor] | None = None,
        rng: torch.Tensor | None = None,
        rng_genot: torch.Tensor | None = None,
        batched: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, TensorLike] | dict[str, torch.Tensor]:
        """Generate the push-forward of ``x`` under condition ``condition``.

        This function solves the ODE learnt with
        the :class:`~cellflow.networks.ConditionalVelocityField`.

        Parameters
        ----------
        x
            Input data of shape [batch_size, ...].
        condition
            Condition of the input data of shape [batch_size, ...].
        rng
            Random number generator to sample from the latent distribution,
            only used if ``condition_mode='stochastic'``. If :obj:`None`, the
            mean embedding is used.
        rng_genot
            Random generate used to sample from the latent distribution in cell space.
        batched
            Whether to use batched prediction. This is only supported if the input has
            the same number of cells for each condition. For example, this works when using
            :class:`~cellflow.data.ValidationSampler` to sample the validation data.
        kwargs
            Keyword arguments for :func:`diffrax.diffeqsolve`.

        Returns
        -------
        The push-forward distribution of ``x`` under condition ``condition``.
        """
        if batched and not x:
            return {}

        if batched:
            if isinstance(x, dict):
                keys = sorted(x.keys())
            if condition is not None:
                condition_keys = sorted(set().union(*(condition[k].keys() for k in keys)))

            # assert that the number of cells is the same for each condition
            n_cells = x[keys[0]].shape[0]
            for k in keys:
                assert x[k].shape[0] == n_cells, "The number of cells must be the same for each condition"
            src_inputs = torch.stack([x[k] for k in keys], axis=0)
            batched_conditions = {}
            for cond_key in condition_keys:
                batched_conditions[cond_key] = torch.stack([condition[k][cond_key] for k in keys])
            pred_targets = self._predict_step(x=src_inputs, condition=batched_conditions)
            return {k: pred_targets[i] for i, k in enumerate(keys)}
        else:
            if isinstance(x, dict):
                raise TypeError(f"State `x` should be of type `torch.Tensor` and not `{type(x)}`")
            x_pred = self._predict_step(x, condition)
            return x_pred

    def _predict_step(self, x, condition, time=None, **kwargs) -> torch.Tensor:
        if not time:
            time = torch.linspace(0.0, 1.0, 101)

        x_pred = self.solver.solve(
            self.vf,
            source=x,
            time=time,
            vf_kwargs={"condition_dict": condition},
            rtol=1e-6,
            atol=1e-7,
            return_trajectory=False,
        )
        print(x_pred.shape)
        return x_pred
