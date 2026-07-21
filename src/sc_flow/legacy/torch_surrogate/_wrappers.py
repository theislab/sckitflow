import abc
from collections.abc import Callable

import torch

__all__ = ["BaseSurrogateWrapper", "GenerativeFlowSurrogateWrapper"]


class BaseSurrogateWrapper(abc.ABC, torch.nn.Module):
    """Base class for differentiable wrappers around trained models.

    Presents a trained model as a differentiable ``x -> response`` map so it can be
    used as a *surrogate* in an inverse problem: the input ``x`` (e.g. continuous
    condition covariates) is optimized by gradient descent to steer the response
    toward a target, while the wrapped model stays frozen. Subclasses implement
    :meth:`forward`.
    """

    def __init__(self, model: torch.nn.Module, *args) -> None:
        """Initializes the wrapper from the given model.

        :param model: The surrogate model to wrap around.
        :type model: class: `torch.nn.Module`
        """
        super().__init__()
        self._model = model

    def _write_input_to_step_data(self, x: torch.Tensor) -> torch.Tensor:
        """Adapt the optimized input ``x`` into the condition the wrapped model expects.

        The identity by default. Override (or pass ``condition_adapter``) to map the
        variable being optimized onto whatever the model consumes as its condition.

        :param x: The input tensor being optimized.
        :type x: class: `torch.Tensor`
        """
        return x

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates the model from the input continuous condition covariates.

        :param x: Tensor containing the continuous conditions covariates which to evaluate the surrogate model on.
        :type x: class: `torch.Tensor`
        """


class GenerativeFlowSurrogateWrapper(BaseSurrogateWrapper):
    r"""Surrogate that pushes fixed source cells through a trained flow's ODE.

    Wraps a trained conditional velocity field ``v(t, state, condition)`` and, given
    a batch of ``N`` candidate conditions ``x`` of shape ``(N, cond_dim)``, integrates
    the flow ODE (explicit Euler) forward from each of the ``M`` fixed source cells
    under each candidate condition. The result is the pushed-forward response of shape
    ``(M, N, G)`` — ``M`` cells, ``N`` candidates, ``G`` features — which
    :class:`~sc_flow.backends.torch.surrogate._base.SurrogatePotential` reduces over
    the cell axis (``reduction_input='mean'``) to a per-candidate response.

    The whole integration is differentiable in ``x``, so an inverse problem can
    backpropagate the potential's gradient into the candidate conditions while the
    wrapped velocity field stays frozen.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        source: torch.Tensor,
        *,
        n_steps: int = 20,
        t0: float = 0.0,
        t1: float = 1.0,
        condition_adapter: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """Initializes the wrapper from the given model and source cells.

        :param model: The trained velocity field, callable as ``model(t, state, condition)``.
        :type model: class: `torch.nn.Module`

        :param source: The ``(M, D)`` source cells to push forward, held fixed across the
            optimization and registered as a buffer.
        :type source: class: `torch.Tensor`

        :param n_steps: Number of explicit-Euler integration steps, defaults to `20`.
        :type n_steps: class: `int`

        :param t0: Integration start time, defaults to `0.0`.
        :type t0: class: `float`

        :param t1: Integration end time, defaults to `1.0`.
        :type t1: class: `float`

        :param condition_adapter: Optional map from the optimized ``x`` to the condition
            the model expects (see :meth:`_write_input_to_step_data`), defaults to `None`.
        :type condition_adapter: `Callable[[torch.Tensor], torch.Tensor] | None`
        """
        super().__init__(model)
        self.register_buffer("_source", source)
        self._n_steps = n_steps
        self._t0 = t0
        self._t1 = t1
        self._condition_adapter = condition_adapter

    def _write_input_to_step_data(self, x: torch.Tensor) -> torch.Tensor:
        if self._condition_adapter is None:
            return x
        return self._condition_adapter(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Push the source cells through the flow under each candidate condition ``x``.

        :param x: Candidate conditions of shape ``(N, cond_dim)``.
        :type x: class: `torch.Tensor`

        :return: The pushed-forward cells of shape ``(M, N, G)``.
        :rtype: class: `torch.Tensor`
        """
        cond = self._write_input_to_step_data(x)
        m, n, d = self._source.shape[0], x.shape[0], self._source.shape[-1]
        # (M, N, D): every source cell evolved under every candidate condition.
        state = self._source.unsqueeze(1).expand(m, n, d).reshape(m * n, d)
        cond_rep = cond.unsqueeze(0).expand(m, n, cond.shape[-1]).reshape(m * n, cond.shape[-1])
        dt = (self._t1 - self._t0) / self._n_steps
        t = state.new_full((m * n, 1), self._t0)
        for _ in range(self._n_steps):
            state = state + dt * self._model(t, state, cond_rep)
            t = t + dt
        return state.reshape(m, n, d)
