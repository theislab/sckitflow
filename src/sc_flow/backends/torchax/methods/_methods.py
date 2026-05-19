import functools
from collections.abc import Callable
from typing import Any

import jax
import optax
import torch
import torchax.interop as interop
from torchax.interop import JittableModule, call_jax, jax_value_and_grad, jax_view, torch_view

import sc_flow.methods as basemethods
from sc_flow import _constants
from sc_flow.backends.torchax.nn._vf import MLPUnconditionalVF
from sc_flow.backends.torchax.probability_paths._probability_paths import BaseProbabilityPath
from sc_flow.backends.torchax.solvers._ode_solver import WrappedODESolver

__all__ = ["FlowMatching"]


class FlowMatching(basemethods.FlowMatching):
    """Torchax flow matching method.

    Keeps a torch-facing API while running all heavy computation on JAX.
    The training step — forward pass, MSE loss, gradient computation, and
    optax parameter update — is compiled end-to-end by XLA via
    :func:`torchax.interop.jax_jit`.

    Parameters
    ----------
    vf
        Torchax velocity field. Must already live on the ``"jax"`` device
        (i.e. be an instance of
        :class:`~sc_flow.backends.torchax.nn.MLPUnconditionalVF`).
    probability_path
        Torchax probability path used to compute ``x_t`` and ``u_t``.
    time_sampler
        Callable with signature ``(rng, n) -> t`` that samples time indices.
    optimizer
        An :mod:`optax` gradient transform (e.g. ``optax.adam(1e-3)``).
        Defaults to ``optax.adam(1e-3)``.
    ema
        Exponential moving average decay for inference weights.
        ``1.0`` disables EMA (inference uses the live weights). Default ``1.0``.
    generate_from_noise
        When ``True``, source samples are replaced with standard-normal noise.
    """

    def __init__(
        self,
        vf: MLPUnconditionalVF,
        probability_path: BaseProbabilityPath,
        time_sampler: Callable,
        optimizer: optax.GradientTransformation | None = None,
        ema: float = 1.0,
        generate_from_noise: bool = True,
    ) -> None:
        super().__init__(
            vf=vf,
            probability_path=probability_path,
            time_sampler=time_sampler,
            ema=ema,
            generate_from_noise=generate_from_noise,
        )

        # --- functional model components ---
        _jittable = JittableModule(vf)
        self._weights = _jittable.params  # dict[str, TorchaxTensor]
        self._buffers = _jittable.buffers  # dict[str, TorchaxTensor] (empty for plain MLPs)
        self._model_fn = functools.partial(_jittable.functional_call, "forward")

        # EMA weights start identical to live weights
        self._weights_ema = self._weights

        # --- optax optimizer ---
        if optimizer is None:
            optimizer = optax.adam(1e-3)
        self._optimizer = optimizer
        self._opt_state = interop.call_jax(optimizer.init, self._weights)

        # --- JIT-compiled training step ---
        # donate_argnums=(0, 1): XLA may reuse the memory of weights and
        # opt_state since they are fully replaced at every step.
        self._train_step_jit = interop.jax_jit(
            self._make_train_step(),
            kwargs_for_jax_jit={"donate_argnums": (0, 1)},
        )

    def _make_train_step(self) -> Callable:
        """Return the pure training-step function to be JIT-compiled.

        The returned function has signature::

            (weights, opt_state, t, xt, ut, condition)
                -> (loss, new_weights, new_opt_state)

        ``xt`` and ``ut`` are pre-computed outside the JIT boundary in
        :meth:`step_fn` so that the PRNG split for the probability path
        does not need to be traced.
        """
        model_fn = self._model_fn
        buffers = self._buffers
        optimizer = self._optimizer

        def train_step(weights, opt_state, t, xt, ut, condition):
            def compute_loss(weights):
                vt = model_fn(weights, buffers, t=t, x=xt, condition_dict={"condition_dict": condition})
                return torch.nn.functional.mse_loss(vt, ut)

            grad_fn = jax_value_and_grad(compute_loss)
            loss, grads = grad_fn(weights)
            updates, opt_state = call_jax(optimizer.update, grads, opt_state)
            new_weights = call_jax(optax.apply_updates, weights, updates)

            return loss, new_weights, opt_state

        return train_step

    def _ema_update(self) -> None:
        """Update the EMA weight snapshot in-place."""
        self._weights_ema = jax.tree_util.tree_map(
            lambda w_ema, w: self.ema * w_ema + (1.0 - self.ema) * w,
            self._weights_ema,
            self._weights,
        )

    def step_fn(self, rng: jax.Array, batch: dict[str, Any]) -> float:
        """Run one training step and return the scalar loss.

        Parameters
        ----------
        rng
            JAX PRNG key for this step.
        batch
            Dict with keys :attr:`~sc_flow._constants.SOURCE_STATE`,
            :attr:`~sc_flow._constants.TARGET_STATE`, and optionally
            ``"condition"``.
        """
        source = batch[_constants.SOURCE_STATE].to("jax")
        target = batch[_constants.TARGET_STATE].to("jax")
        condition = batch.get(_constants.CONDITION_KEY)
        if condition is not None:
            condition = {k: v.to("jax") for k, v in condition.items()}

        rng_time, rng_noise, rng_flow = jax.random.split(rng, 3)
        n = target.shape[0]
        t = self.time_sampler(rng_time, n)

        if self.generate_from_noise:
            tgt_jax = jax_view(target)
            source = torch_view(jax.random.normal(rng_noise, shape=tgt_jax.shape, dtype=tgt_jax.dtype))

        xt = self.probability_path.compute_xt(t, source, target, prng=rng_flow)
        ut = self.probability_path.compute_ut(t, xt, source, target)

        loss, self._weights, self._opt_state = self._train_step_jit(
            self._weights, self._opt_state, t, xt, ut, condition
        )

        if self.ema != 1.0:
            self._ema_update()
        else:
            self._weights_ema = self._weights

        return loss.item()

    def train_step(self, batch: dict[str, Any], prng: jax.Array) -> float:
        """Trainer-compatible wrapper around :meth:`step_fn`."""
        return self.step_fn(prng, batch)

    def predict(
        self,
        source: Any,
        t0: float = 0.0,
        t1: float = 1.0,
        *,
        num_time_steps: int = 500,
        return_trajectory: bool = False,
        condition: dict[str, Any] | None = None,
        solver: "WrappedODESolver | None" = None,
        solver_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        weights_ema = self._weights_ema
        model_fn = self._model_fn
        buffers = self._buffers
        condition_jax = {k: v.to("jax") for k, v in condition.items()} if condition is not None else None

        class _EMAVFWrapper(torch.nn.Module):
            def forward(self, t: Any, x: Any, condition_dict: Any = None) -> Any:
                return model_fn(
                    weights_ema,
                    buffers,
                    t=t,
                    x=x,
                    condition_dict={"condition_dict": condition_dict},
                )

            def get_vf_fn(self, **vf_kwargs: Any):
                """Return a plain callable ``(t, x) -> vt`` as expected by the solver wrappers."""
                _condition = condition_jax

                def vf_fn(t: Any, x: Any) -> Any:
                    return self.forward(t, x, condition_dict=_condition)

                return vf_fn

        ema_vf = _EMAVFWrapper()

        if solver is None:
            solver = WrappedODESolver(dynamics=ema_vf, method="euler")

        source_jax = source.to("jax")
        return solver.solve(
            source_jax,
            t0,
            t1,
            num_time_steps=num_time_steps,
            return_trajectory=return_trajectory,
            solver_kwargs=solver_kwargs,
        )
