import abc

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

    def compute_alpha_t_and_dot_alpha_t(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Returns :math:`\alpha_t` and its time-derivative :math:`\dot{\alpha}_t`.

        :param t: Time tensor, with values in :math:`[0, 1]`.
        :type t: class: `torch.Tensor`

        :returns: The pair :math:`(\alpha_t, \dot{\alpha}_t)`.
        :rtype: class: `tuple` [class: `torch.Tensor`, class: `torch.Tensor`]
        """
        alpha_t, grad_fn = torch.func.vjp(self.compute_alpha_t, t)
        return alpha_t, grad_fn(torch.ones_like(alpha_t))[0]

    def compute_beta_t_and_dot_beta_t(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Returns :math:`\beta_t` and its time-derivative :math:`\dot{\beta}_t`.

        :param t: Time tensor, with values in :math:`[0, 1]`.
        :type t: class: `torch.Tensor`

        :returns: The pair :math:`(\beta_t, \dot{\beta}_t)`.
        :rtype: class: `tuple` [class: `torch.Tensor`, class: `torch.Tensor`]
        """
        beta_t, grad_fn = torch.func.vjp(self.compute_beta_t, t)
        return beta_t, grad_fn(torch.ones_like(beta_t))[0]

    def compute_sigma_t_and_dot_sigma_t(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Returns :math:`\sigma_t` and its time-derivative :math:`\dot{\sigma}_t`.

        :param t: Time tensor, with values in :math:`[0, 1]`.
        :type t: class: `torch.Tensor`

        :returns: The pair :math:`(\sigma_t, \dot{\sigma}_t)`.
        :rtype: class: `tuple` [class: `torch.Tensor`, class: `torch.Tensor`]
        """
        sigma_t, grad_fn = torch.func.vjp(self.compute_sigma_t, t)
        return sigma_t, grad_fn(torch.ones_like(sigma_t))[0]

    def compute_dot_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        r"""Returns the time-derivative :math:`\dot{\alpha}_t`.

        :param t: Time tensor, with values in :math:`[0, 1]`.
        :type t: class: `torch.Tensor`

        :returns: The derivative :math:`\dot{\alpha}_t`.
        :rtype: class: `torch.Tensor`
        """
        return self.compute_alpha_t_and_dot_alpha_t(t)[1]

    def compute_dot_beta_t(self, t: torch.Tensor) -> torch.Tensor:
        r"""Returns the time-derivative :math:`\dot{\beta}_t`.

        :param t: Time tensor, with values in :math:`[0, 1]`.
        :type t: class: `torch.Tensor`

        :returns: The derivative :math:`\dot{\beta}_t`.
        :rtype: class: `torch.Tensor`
        """
        return self.compute_beta_t_and_dot_beta_t(t)[1]

    def compute_dot_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        r"""Returns the time-derivative :math:`\dot{\sigma}_t`.

        :param t: Time tensor, with values in :math:`[0, 1]`.
        :type t: class: `torch.Tensor`

        :returns: The derivative :math:`\dot{\sigma}_t`.
        :rtype: class: `torch.Tensor`
        """
        return self.compute_sigma_t_and_dot_sigma_t(t)[1]

    def compute_xt(
        self,
        t: torch.Tensor,
        xt: torch.Tensor,
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

        :param xt: Unused; kept for signature symmetry with
            :meth:`compute_ut`.
        :type xt: class: `torch.Tensor`

        :param x0: The initial sample :math:`\boldsymbol{x}_0`.
        :type x0: class: `torch.Tensor`

        :param x1: The final sample :math:`\boldsymbol{x}_1`.
        :type x1: class: `torch.Tensor`

        :param generator: Random generator used to draw the noise.
        :type generator: class: `torch.Generator`

        :returns: The sample :math:`\boldsymbol{x}_t`.
        :rtype: class: `torch.Tensor`
        """
        alpha_t = broadcast_to_target_shape(self.compute_alpha_t(t), x0.shape)
        beta_t = broadcast_to_target_shape(self.compute_beta_t(t), x0.shape)
        mu_t = alpha_t * x0 + beta_t * x1
        if self._sigma > 0:
            sigma_t = broadcast_to_target_shape(self.compute_sigma_t(t), x0.shape)
            noise = self._sampler(x1.shape, generator=generator, device=x0.device, dtype=x0.dtype)
            return mu_t + sigma_t * noise
        return mu_t

    def compute_ut(
        self,
        t: torch.Tensor,
        xt: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
        *,
        generator: torch.Generator,
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

        :param generator: Unused; kept for signature symmetry with
            :meth:`compute_xt`.
        :type generator: class: `torch.Generator`

        :returns: The velocity :math:`\boldsymbol{u}_t`.
        :rtype: class: `torch.Tensor`
        """
        # Compute alpha terms.
        alpha_t, dot_alpha_t = self.compute_alpha_t_and_dot_alpha_t(t)
        alpha_t = broadcast_to_target_shape(alpha_t, x0.shape)
        dot_alpha_t = broadcast_to_target_shape(dot_alpha_t, x0.shape)

        # Compute beta terms.
        beta_t, dot_beta_t = self.compute_beta_t_and_dot_beta_t(t)
        beta_t = broadcast_to_target_shape(beta_t, x0.shape)
        dot_beta_t = broadcast_to_target_shape(dot_beta_t, x0.shape)

        # Compute mean and differential.
        mu_t = alpha_t * x0 + beta_t * x1
        dot_mu_t = dot_alpha_t * x0 + dot_beta_t * x1

        # Return mean differential for deterministic paths.
        if self._sigma <= 0:
            return dot_mu_t

        # Compute sigma terms.
        sigma_t, dot_sigma_t = self.compute_sigma_t_and_dot_sigma_t(t)
        sigma_t = broadcast_to_target_shape(sigma_t, x0.shape)
        dot_sigma_t = broadcast_to_target_shape(dot_sigma_t, x0.shape)

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

    def compute_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        r"""Returns :math:`\alpha_t \equiv 0` for every :math:`t`.

        :param t: Time tensor, with values in :math:`[0, 1]`.
        :type t: class: `torch.Tensor`

        :returns: A tensor of zeros with the same shape as ``t``.
        :rtype: class: `torch.Tensor`
        """
        return torch.zeros_like(t)
