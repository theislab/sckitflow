import torch

from sckitflow._constants import PI
from sckitflow.core.probability_paths._base import BasePath, OneSidedPath

__all__ = [
    "LinearTwoSidedPath",
    "TrigTwoSidedPath",
    "EncDecTwoSidedPath",
    "LinearOneSidedPath",
    "TrigOneSidedPath",
    "VPOneSidedPath",
]


class _LinearPathMixin:
    r"""Mixin supplying a linear :math:`\beta_t = t` coefficient.

    Combined with :class:`OneSidedPath` (where :math:`\alpha_t \equiv 0`)
    this gives paths whose mean grows linearly from the origin at
    :math:`t = 0` to :math:`\boldsymbol{x}_1` at :math:`t = 1`.
    """

    def compute_beta_t(self, t: torch.Tensor) -> torch.Tensor:
        r"""Returns :math:`\beta_t = t`.

        :param t: Time tensor, with values in :math:`[0, 1]`.
        :type t: class: `torch.Tensor`

        :returns: The coefficient :math:`t`, with the same shape as ``t``.
        :rtype: class: `torch.Tensor`
        """
        return t


class _TrigPathMixin:
    r"""Mixin supplying a sinusoidal :math:`\beta_t = \sin\!\left(\tfrac{\pi}{2} t\right)` coefficient.

    The derivative vanishes at both endpoints, which gives smoother path
    endpoints than :class:`_LinearPathMixin`.
    """

    def compute_beta_t(self, t: torch.Tensor) -> torch.Tensor:
        r"""Returns :math:`\beta_t = \sin\!\left(\tfrac{\pi}{2} t\right)`.

        :param t: Time tensor, with values in :math:`[0, 1]`.
        :type t: class: `torch.Tensor`

        :returns: The coefficient, with the same shape as ``t``.
        :rtype: class: `torch.Tensor`
        """
        return torch.sin(0.5 * PI * t)


class LinearTwoSidedPath(_LinearPathMixin, BasePath):
    r"""Two-sided path with linear interpolation and :math:`\sqrt{\cdot}` noise schedule.

    The interpolation coefficients are

    .. math::

        \alpha_t = 1 - t,
        \qquad
        \beta_t = t,
        \qquad
        \sigma_t = \sqrt{\sigma \, t \, (1 - t)}.

    The path runs from :math:`\boldsymbol{x}_0` at :math:`t = 0` to
    :math:`\boldsymbol{x}_1` at :math:`t = 1`, and the standard deviation
    vanishes at both endpoints.
    """

    def compute_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        r"""Returns :math:`\alpha_t = 1 - t`.

        :param t: Time tensor, with values in :math:`[0, 1]`.
        :type t: class: `torch.Tensor`

        :returns: The coefficient :math:`1 - t`.
        :rtype: class: `torch.Tensor`
        """
        return 1 - t

    def compute_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        r"""Returns :math:`\sigma_t = \sqrt{\sigma \, t \, (1 - t)}`.

        :param t: Time tensor, with values in :math:`[0, 1]`.
        :type t: class: `torch.Tensor`

        :returns: The standard deviation :math:`\sigma_t`.
        :rtype: class: `torch.Tensor`
        """
        return torch.sqrt(self.sigma * t * (1 - t))


class TrigTwoSidedPath(_TrigPathMixin, BasePath):
    r"""Two-sided path with cosine/sine interpolation and :math:`\sqrt{\cdot}` noise schedule.

    The interpolation coefficients are

    .. math::

        \alpha_t = \cos\!\left(\tfrac{\pi}{2} t\right),
        \qquad
        \beta_t = \sin\!\left(\tfrac{\pi}{2} t\right),
        \qquad
        \sigma_t = \sqrt{\sigma \, t \, (1 - t)}.

    The path runs from :math:`\boldsymbol{x}_0` at :math:`t = 0` to
    :math:`\boldsymbol{x}_1` at :math:`t = 1`, with vanishing derivatives
    of :math:`\alpha_t` and :math:`\beta_t` at the endpoints.
    """

    def compute_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        r"""Returns :math:`\alpha_t = \cos\!\left(\tfrac{\pi}{2} t\right)`.

        :param t: Time tensor, with values in :math:`[0, 1]`.
        :type t: class: `torch.Tensor`

        :returns: The coefficient :math:`\alpha_t`.
        :rtype: class: `torch.Tensor`
        """
        return torch.cos(0.5 * PI * t)

    def compute_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        r"""Returns :math:`\sigma_t = \sqrt{\sigma \, t \, (1 - t)}`.

        :param t: Time tensor, with values in :math:`[0, 1]`.
        :type t: class: `torch.Tensor`

        :returns: The standard deviation :math:`\sigma_t`.
        :rtype: class: `torch.Tensor`
        """
        return torch.sqrt(self.sigma * t * (1 - t))


class EncDecTwoSidedPath(BasePath):
    r"""Two-sided path with piecewise :math:`\cos^2(\pi t)` interpolation.

    The path is built from two one-sided segments joined at
    :math:`t = \tfrac{1}{2}`:

    .. math::

        \alpha_t =
        \begin{cases}
            \cos^2(\pi t), & t \le \tfrac{1}{2}, \\
            0,             & t > \tfrac{1}{2},
        \end{cases}
        \qquad
        \beta_t =
        \begin{cases}
            0,             & t \le \tfrac{1}{2}, \\
            \cos^2(\pi t), & t > \tfrac{1}{2}.
        \end{cases}

    The standard deviation is

    .. math::

        \sigma_t = \sigma \sin^2(\pi t),

    which vanishes at both endpoints and peaks at :math:`t = \tfrac{1}{2}`.
    This path is the "encoder/decoder" schedule: the model first collapses
    :math:`\boldsymbol{x}_0` to the origin, then expands towards
    :math:`\boldsymbol{x}_1`.
    """

    def compute_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        r"""Returns :math:`\alpha_t = \cos^2(\pi t)` for :math:`t \le \tfrac{1}{2}`, and :math:`0` otherwise.

        :param t: Time tensor, with values in :math:`[0, 1]`.
        :type t: class: `torch.Tensor`

        :returns: The coefficient :math:`\alpha_t`.
        :rtype: class: `torch.Tensor`
        """
        return torch.where(
            t <= 0.5,
            torch.cos(PI * t) ** 2,
            torch.zeros_like(t),
        )

    def compute_beta_t(self, t: torch.Tensor) -> torch.Tensor:
        r"""Returns :math:`0` for :math:`t \le \tfrac{1}{2}`, and :math:`\beta_t = \cos^2(\pi t)` otherwise.

        :param t: Time tensor, with values in :math:`[0, 1]`.
        :type t: class: `torch.Tensor`

        :returns: The coefficient :math:`\beta_t`.
        :rtype: class: `torch.Tensor`
        """
        return torch.where(
            t > 0.5,
            torch.cos(PI * t) ** 2,
            torch.zeros_like(t),
        )

    def compute_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        r"""Returns :math:`\sigma_t = \sigma \sin^2(\pi t)`.

        :param t: Time tensor, with values in :math:`[0, 1]`.
        :type t: class: `torch.Tensor`

        :returns: The standard deviation :math:`\sigma_t`.
        :rtype: class: `torch.Tensor`
        """
        return self.sigma * torch.sin(PI * t) ** 2


class LinearOneSidedPath(_LinearPathMixin, OneSidedPath):
    r"""One-sided path with linear interpolation and linearly decaying noise.

    .. math::

        \alpha_t = 0,
        \qquad
        \beta_t = t,
        \qquad
        \sigma_t = 1 - t.

    Since :math:`\alpha_t \equiv 0`, the initial sample
    :math:`\boldsymbol{x}_0` is unused; the path draws
    :math:`\boldsymbol{x}_t` from :math:`t \boldsymbol{x}_1` plus noise that
    decays linearly to zero at :math:`t = 1`.
    """

    def compute_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        r"""Returns :math:`\sigma_t = 1 - t`.

        :param t: Time tensor, with values in :math:`[0, 1]`.
        :type t: class: `torch.Tensor`

        :returns: The standard deviation :math:`\sigma_t`.
        :rtype: class: `torch.Tensor`
        """
        return 1 - t


class TrigOneSidedPath(_TrigPathMixin, OneSidedPath):
    r"""One-sided path with sinusoidal interpolation and cosine-decaying noise.

    .. math::

        \alpha_t = 0,
        \qquad
        \beta_t = \sin\!\left(\tfrac{\pi}{2} t\right),
        \qquad
        \sigma_t = \cos\!\left(\tfrac{\pi}{2} t\right).

    The coefficients satisfy :math:`\beta_t^2 + \sigma_t^2 = 1`, so the
    path preserves the variance of unit Gaussian noise.
    """

    def compute_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        r"""Returns :math:`\sigma_t = \cos\!\left(\tfrac{\pi}{2} t\right)`.

        :param t: Time tensor, with values in :math:`[0, 1]`.
        :type t: class: `torch.Tensor`

        :returns: The standard deviation :math:`\sigma_t`.
        :rtype: class: `torch.Tensor`
        """
        return torch.cos(0.5 * PI * t)


class VPOneSidedPath(_LinearPathMixin, OneSidedPath):
    r"""One-sided variance-preserving path.

    .. math::

        \alpha_t = 0,
        \qquad
        \beta_t = t,
        \qquad
        \sigma_t = \sqrt{1 - t^2}.

    The coefficients satisfy :math:`\beta_t^2 + \sigma_t^2 = 1`, so the
    path preserves the variance of unit Gaussian noise; the standard
    deviation vanishes at :math:`t = 1`.
    """

    def compute_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        r"""Returns :math:`\sigma_t = \sqrt{1 - t^2}`.

        :param t: Time tensor, with values in :math:`[0, 1]`.
        :type t: class: `torch.Tensor`

        :returns: The standard deviation :math:`\sigma_t`.
        :rtype: class: `torch.Tensor`
        """
        return torch.sqrt(1 - t**2)
