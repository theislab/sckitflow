import abc
from collections.abc import Callable

import torch

from sckitflow._samplers import NoiseSamplerConfig
from sckitflow.core._utils import broadcast_to_target_shape

__all__ = ["BasePath", "OneSidedPath"]


class BasePath(abc.ABC):
    r"""Base class for conditional probability paths :math:`p_t(\boldsymbol{x}_t | \boldsymbol{x}_0, \boldsymbol{x}_1)`.

    A path interpolates between two endpoints :math:`\boldsymbol{x}_0` at
    :math:`t = 0` and :math:`\boldsymbol{x}_1` at :math:`t = 1` through a
    time-dependent mean

    .. math::

        \boldsymbol{\mu}_t = \alpha_t \, \boldsymbol{x}_0
                           + \beta_t \, \boldsymbol{x}_1

    and a time-dependent standard deviation :math:`\sigma_t`. Samples along
    the path are drawn as

    .. math::

        \boldsymbol{x}_t = \boldsymbol{\mu}_t
                         + \sigma_t \, \boldsymbol{\epsilon},
        \qquad \boldsymbol{\epsilon} \sim \mathcal{N}(0, I).

    Subclasses supply :math:`\alpha_t`, :math:`\beta_t` and
    :math:`\sigma_t` by implementing :meth:`compute_alpha_t`,
    :meth:`compute_beta_t` and :meth:`compute_sigma_t`; the remaining
    machinery (sampling, time-derivatives, velocity field) is shared here.
    """

    def __init__(
        self,
        sigma: float,
        sampler_cfg: NoiseSamplerConfig | None = None,
    ) -> None:
        r"""Initializes the probability path.

        :param sigma: Non-negative scalar noise strength. It scales the
            time-dependent standard deviation :math:`\sigma_t` of the
            conditional probability path; a value of :math:`0` yields a
            deterministic path.
        :type sigma: class: `float`

        :param sampler_cfg: Configuration used to build the noise sampler.
            Required when ``sigma > 0`` and ignored otherwise.
        :type sampler_cfg: class: `NoiseSamplerConfig` | None

        :raises ValueError: If ``sigma`` is negative, or if ``sigma > 0``
            is given without a ``sampler_cfg``.
        """
        # sanity check for positive values
        if not sigma >= 0:
            msg = f"Argument sigma should be a positive float for non deterministic probability paths. Found {sigma=}"
            raise ValueError(msg)
        if sigma > 0 and sampler_cfg is None:
            raise ValueError("Positive sigma requires a sampler to be passed")

        # setting the attributes
        self._sigma = sigma
        self._sampler = None if sampler_cfg is None else sampler_cfg.build()

    @abc.abstractmethod
    def compute_alpha_t(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        r"""Returns the coefficient :math:`\alpha_t` of the initial sample.

        :param t: Time tensor, with values in :math:`[0, 1]`.
        :type t: class: `torch.Tensor`

        :returns: The coefficient :math:`\alpha_t`.
        :rtype: class: `torch.Tensor`
        """

    @abc.abstractmethod
    def compute_beta_t(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        r"""Returns the coefficient :math:`\beta_t` of the final sample.

        :param t: Time tensor, with values in :math:`[0, 1]`.
        :type t: class: `torch.Tensor`

        :returns: The coefficient :math:`\beta_t`.
        :rtype: class: `torch.Tensor`
        """

    @abc.abstractmethod
    def compute_sigma_t(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        r"""Returns the time-dependent standard deviation :math:`\sigma_t`.

        :param t: Time tensor, with values in :math:`[0, 1]`.
        :type t: class: `torch.Tensor`

        :returns: The standard deviation :math:`\sigma_t`.
        :rtype: class: `torch.Tensor`
        """

    def _value_and_dot(
        self, fn: Callable[[torch.Tensor], torch.Tensor], t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Returns ``fn(t)`` and its time-derivative, by a VJP so the derivative stays analytic.

        :param fn: One of the coefficient methods.
        :type fn: class: `Callable`

        :param t: Time tensor, with values in :math:`[0, 1]`.
        :type t: class: `torch.Tensor`

        :returns: The pair ``(fn(t), d fn(t) / dt)``.
        :rtype: class: `tuple` [class: `torch.Tensor`, class: `torch.Tensor`]
        """
        value, grad_fn = torch.func.vjp(fn, t)
        return value, grad_fn(torch.ones_like(value))[0]

    def compute_alpha_t_and_dot_alpha_t(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Returns :math:`\alpha_t` and its time-derivative :math:`\dot{\alpha}_t`.

        :param t: Time tensor, with values in :math:`[0, 1]`.
        :type t: class: `torch.Tensor`

        :returns: The pair :math:`(\alpha_t, \dot{\alpha}_t)`.
        :rtype: class: `tuple` [class: `torch.Tensor`, class: `torch.Tensor`]
        """
        return self._value_and_dot(self.compute_alpha_t, t)

    def compute_beta_t_and_dot_beta_t(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Returns :math:`\beta_t` and its time-derivative :math:`\dot{\beta}_t`.

        :param t: Time tensor, with values in :math:`[0, 1]`.
        :type t: class: `torch.Tensor`

        :returns: The pair :math:`(\beta_t, \dot{\beta}_t)`.
        :rtype: class: `tuple` [class: `torch.Tensor`, class: `torch.Tensor`]
        """
        return self._value_and_dot(self.compute_beta_t, t)

    def compute_sigma_t_and_dot_sigma_t(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Returns :math:`\sigma_t` and its time-derivative :math:`\dot{\sigma}_t`.

        :param t: Time tensor, with values in :math:`[0, 1]`.
        :type t: class: `torch.Tensor`

        :returns: The pair :math:`(\sigma_t, \dot{\sigma}_t)`.
        :rtype: class: `tuple` [class: `torch.Tensor`, class: `torch.Tensor`]
        """
        return self._value_and_dot(self.compute_sigma_t, t)

    def compute_xt(
        self,
        t: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        r"""Draws a sample :math:`\boldsymbol{x}_t` along the path.

        The sample is computed as

        .. math::

            \boldsymbol{x}_t = \alpha_t \, \boldsymbol{x}_0
                             + \beta_t \, \boldsymbol{x}_1
                             + \sigma_t \, \boldsymbol{\epsilon},

        where :math:`\boldsymbol{\epsilon}` is drawn from the sampler when
        ``sigma > 0``. For deterministic paths (``sigma == 0``) the noise
        term vanishes and the output equals the mean :math:`\mu_t`.

        :param t: Time tensor, with values in :math:`[0, 1]`.
        :type t: class: `torch.Tensor`

        :param x0: The initial sample :math:`\boldsymbol{x}_0`.
        :type x0: class: `torch.Tensor`

        :param x1: The final sample :math:`\boldsymbol{x}_1`.
        :type x1: class: `torch.Tensor`

        :param generator: Random generator used to draw the noise.
        :type generator: class: `torch.Generator`

        :returns: The sample :math:`\boldsymbol{x}_t`.
        :rtype: class: `torch.Tensor`
        """

        def bc(value: torch.Tensor) -> torch.Tensor:
            return broadcast_to_target_shape(value, x0.shape)

        mu_t = bc(self.compute_alpha_t(t)) * x0 + bc(self.compute_beta_t(t)) * x1
        if self._sigma > 0:
            noise = self._sampler(x1.shape, generator=generator, device=x0.device, dtype=x0.dtype)
            return mu_t + bc(self.compute_sigma_t(t)) * noise
        return mu_t

    def compute_ut(
        self,
        t: torch.Tensor,
        xt: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        r"""Returns the velocity field :math:`\boldsymbol{u}_t` along the path.

        The velocity is computed analytically from the interpolation
        coefficients as

        .. math::

            \boldsymbol{u}_t = \dot{\boldsymbol{\mu}}_t
                             + \frac{\dot{\sigma}_t}{\sigma_t}
                               \left( \boldsymbol{x}_t - \boldsymbol{\mu}_t \right),

        with

        .. math::

            \boldsymbol{\mu}_t = \alpha_t \, \boldsymbol{x}_0
                               + \beta_t \, \boldsymbol{x}_1,
            \qquad
            \dot{\boldsymbol{\mu}}_t = \dot{\alpha}_t \, \boldsymbol{x}_0
                                     + \dot{\beta}_t \, \boldsymbol{x}_1.

        For deterministic paths (``sigma == 0``) the noise term is dropped
        and the velocity equals :math:`\dot{\boldsymbol{\mu}}_t`.

        :param t: Time tensor, with values in :math:`[0, 1]`.
        :type t: class: `torch.Tensor`

        :param xt: The sample :math:`\boldsymbol{x}_t` at time ``t``.
        :type xt: class: `torch.Tensor`

        :param x0: The initial sample :math:`\boldsymbol{x}_0`.
        :type x0: class: `torch.Tensor`

        :param x1: The final sample :math:`\boldsymbol{x}_1`.
        :type x1: class: `torch.Tensor`

        :returns: The velocity :math:`\boldsymbol{u}_t`.
        :rtype: class: `torch.Tensor`
        """

        def bc(pair: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            value, dot_value = pair
            return broadcast_to_target_shape(value, x0.shape), broadcast_to_target_shape(dot_value, x0.shape)

        alpha_t, dot_alpha_t = bc(self.compute_alpha_t_and_dot_alpha_t(t))
        beta_t, dot_beta_t = bc(self.compute_beta_t_and_dot_beta_t(t))

        # Compute mean and differential.
        mu_t = alpha_t * x0 + beta_t * x1
        dot_mu_t = dot_alpha_t * x0 + dot_beta_t * x1

        # Return mean differential for deterministic paths.
        if self._sigma <= 0:
            return dot_mu_t

        sigma_t, dot_sigma_t = bc(self.compute_sigma_t_and_dot_sigma_t(t))

        # Guard for zero division
        eps = torch.finfo(sigma_t.dtype).eps
        safe_sigma_t = torch.where(sigma_t.abs() > eps, sigma_t, torch.ones_like(sigma_t))
        coeff = torch.where(
            sigma_t.abs() > eps,
            dot_sigma_t / safe_sigma_t,
            torch.zeros_like(sigma_t),
        )
        return dot_mu_t + coeff * (xt - mu_t)

    @property
    def sigma(self) -> float:
        r"""The scalar noise strength :math:`\sigma` of the path.

        :rtype: class: `float`
        """
        return self._sigma


class OneSidedPath(BasePath):
    r"""Base class for one-sided probability paths.

    One-sided paths interpolate only towards :math:`\boldsymbol{x}_1`; the
    coefficient :math:`\alpha_t` is identically zero, so the initial sample
    :math:`\boldsymbol{x}_0` is never used. Subclasses only need to supply
    :math:`\beta_t` and :math:`\sigma_t`.
    """

    def __init__(self, sigma: float, sampler_cfg: NoiseSamplerConfig | None = None) -> None:
        r"""Initializes the one-sided path.

        :raises ValueError: If ``sigma`` is zero. With :math:`\alpha_t \equiv 0` the noise term is the only thing
            standing in for :math:`\boldsymbol{x}_0`, so a deterministic one-sided path would collapse
            :math:`\boldsymbol{x}_t` to :math:`\beta_t \boldsymbol{x}_1` and flow from the origin.
        """
        if sigma == 0:
            raise ValueError(
                f"A one-sided path interpolates noise into the data, so it needs sigma > 0; found {sigma=}. "
                "Use a two-sided path for a deterministic interpolation between two samples."
            )
        super().__init__(sigma, sampler_cfg)

    def compute_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        r"""Returns :math:`\alpha_t \equiv 0` for every :math:`t`.

        :param t: Time tensor, with values in :math:`[0, 1]`.
        :type t: class: `torch.Tensor`

        :returns: A tensor of zeros with the same shape as ``t``.
        :rtype: class: `torch.Tensor`
        """
        return torch.zeros_like(t)
