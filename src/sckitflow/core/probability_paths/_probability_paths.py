import torch

from sckitflow._constants import PI
from sckitflow._samplers import NoiseSamplerConfig
from sckitflow.core.probability_paths._base import BasePath

__all__ = [
    "LinearPath",
    "TrigPath",
    "EncDecPath",
]


class _BridgeNoiseMixin:
    r"""Mixin supplying the Brownian-bridge noise schedule :math:`\sigma_t = \sigma \sqrt{t (1 - t)}`.

    The standard deviation vanishes at both endpoints and peaks at :math:`t = \tfrac{1}{2}`, so the path is
    anchored on :math:`\boldsymbol{x}_0` and :math:`\boldsymbol{x}_1` and noisiest in between.
    """

    def compute_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        r"""Returns :math:`\sigma_t = \sigma \sqrt{t \, (1 - t)}`."""
        return self.sigma * torch.sqrt(t * (1 - t))


class LinearPath(_BridgeNoiseMixin, BasePath):
    r"""Path with linear interpolation and :math:`\sqrt{\cdot}` noise schedule.

    The interpolation coefficients are

    .. math::

        \alpha_t = 1 - t,
        \qquad
        \beta_t = t,
        \qquad
        \sigma_t = \sigma \sqrt{t \, (1 - t)}.

    The path runs from :math:`\boldsymbol{x}_0` at :math:`t = 0` to
    :math:`\boldsymbol{x}_1` at :math:`t = 1`, and the standard deviation
    vanishes at both endpoints.
    """

    def compute_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        r"""Returns :math:`\alpha_t = 1 - t`."""
        return 1 - t

    def compute_beta_t(self, t: torch.Tensor) -> torch.Tensor:
        r"""Returns :math:`\beta_t = t`."""
        return t


class TrigPath(_BridgeNoiseMixin, BasePath):
    r"""Path with cosine/sine interpolation and :math:`\sqrt{\cdot}` noise schedule.

    The interpolation coefficients are

    .. math::

        \alpha_t = \cos\!\left(\tfrac{\pi}{2} t\right),
        \qquad
        \beta_t = \sin\!\left(\tfrac{\pi}{2} t\right),
        \qquad
        \sigma_t = \sigma \sqrt{t \, (1 - t)}.

    The path runs from :math:`\boldsymbol{x}_0` at :math:`t = 0` to
    :math:`\boldsymbol{x}_1` at :math:`t = 1`. With :math:`\sigma = 0` and :math:`\boldsymbol{x}_0` and :math:`\boldsymbol{x}_1` independent with unit
    variance, :math:`\alpha_t^2 + \beta_t^2 = 1` keeps the variance constant along the path.
    """

    def compute_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        r"""Returns :math:`\alpha_t = \cos\!\left(\tfrac{\pi}{2} t\right)`."""
        return torch.cos(0.5 * PI * t)

    def compute_beta_t(self, t: torch.Tensor) -> torch.Tensor:
        r"""Returns :math:`\beta_t = \sin\!\left(\tfrac{\pi}{2} t\right)`."""
        return torch.sin(0.5 * PI * t)


class EncDecPath(BasePath):
    r"""Path with piecewise :math:`\cos^2(\pi t)` interpolation.

    The path is built from two half-paths joined at
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
    :math:`\boldsymbol{x}_1`. At :math:`t = \tfrac{1}{2}` both :math:`\alpha_t` and :math:`\beta_t` vanish,
    so the path needs ``sigma > 0``.
    """

    def __init__(self, sigma: float, sampler_cfg: NoiseSamplerConfig | None = None) -> None:
        if sigma == 0:
            raise ValueError("EncDecPath needs sigma > 0: with sigma == 0 every x_t is 0 at t = 0.5.")
        super().__init__(sigma, sampler_cfg=sampler_cfg)

    def compute_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        r"""Returns :math:`\alpha_t = \cos^2(\pi t)` for :math:`t \le \tfrac{1}{2}`, and :math:`0` otherwise."""
        return torch.where(
            t <= 0.5,
            torch.cos(PI * t) ** 2,
            torch.zeros_like(t),
        )

    def compute_beta_t(self, t: torch.Tensor) -> torch.Tensor:
        r"""Returns :math:`0` for :math:`t \le \tfrac{1}{2}`, and :math:`\beta_t = \cos^2(\pi t)` otherwise."""
        return torch.where(
            t > 0.5,
            torch.cos(PI * t) ** 2,
            torch.zeros_like(t),
        )

    def compute_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        r"""Returns :math:`\sigma_t = \sigma \sin^2(\pi t)`."""
        return self.sigma * torch.sin(PI * t) ** 2
