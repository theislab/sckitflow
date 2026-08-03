import abc
import logging

import torch

from sc_flow._constants import PI
from sc_flow.flow._torch_utils import broadcast_to_target_shape

logger = logging.getLogger(__name__)

__all__ = [
    "BaseProbabilityPath",
    "LinearProbabilityPath",
    "LinearGaussianProbabilityPath",
    "SchrodingerBridgeProbabilityPath",
    "LinearDiracProbabilityPath",
    "VariancePreservingDiracProbabilityPath",
]


class BaseProbabilityPath(abc.ABC):

    _require_prng: bool

    def __init__(
        self,
        sigma: float,
        prng: torch.Generator | None = None,
    ) -> None:
        # sanity check when we require the prng
        if self._require_prng and (not isinstance(prng, torch.Generator)):
            msg = (
                f"The probability path  {self.__class__.__name__} requires a PRNG. Please provide an instance of `torch.Generator`"
                r"for reproducible results. Setting it to \`torch.random.default_generator\` by default."
            )
            logger.warning(msg)
            prng = torch.random.default_generator
        elif (not self._require_prng) and (prng is not None):
            msg = f"PRNG provided to {self.__class__.__name__}, which is deterministic. Setting it to `None`."
            logger.warning(msg)
            prng = None
        # sanity check for positive values
        if not sigma > 0 and not self.is_deterministic:
            msg = f"Argument sigma should be a positive float for non deterministic probability paths. Found {sigma=}"
            raise ValueError(msg)

        # setting the attributes
        self._prng = prng
        self._sigma = sigma

    def _verify_shapes(
        self,
        input_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
    ) -> None:
        if input_tensor.shape != target_tensor.shape:
            msg = (
                "`input_tensor` and `target_tensor` are supposed to have the same shape"
                f"found  {input_tensor.shape=} and {target_tensor.shape=}"
            )
            raise ValueError(msg)

    @abc.abstractmethod
    def compute_mu_t(
        self,
        t: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def compute_sigma_t(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def compute_ut(
        self,
        t: torch.Tensor,
        xt: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        ...

    def compute_xt(
        self,
        t: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        # handling shapes
        t = broadcast_to_target_shape(t, x0.shape)
        self._verify_shapes(x0, x1)
        self._verify_shapes(t, x1)
        # computing coefficients and noise value
        mu_t = self.compute_mu_t(t, x0, x1)
        # sampling noise
        if self._require_prng:
            sigma_t = self.compute_sigma_t(t)
            noise = torch.randn(*x0.shape, generator=self._prng).to(device=x0.device, dtype=x0.dtype)
            return mu_t + sigma_t * noise
        # returning mean for deterministic paths
        return mu_t

    @classmethod
    @property
    def is_deterministic(
        cls,
    ) -> bool:
        return not cls._require_prng


class LinearProbabilityPath(BaseProbabilityPath, abc.ABC):

    _require_prng: bool

    def __init__(
        self,
        sigma: float,
        prng: torch.Generator | None = None,
    ) -> None:
        super().__init__(sigma=sigma, prng=prng)

    def compute_mu_t(
        self,
        t: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        # handling shapes
        t = broadcast_to_target_shape(t, x0.shape)
        self._verify_shapes(x0, x1)
        self._verify_shapes(t, x1)

        return (1 - t) * x0 + t * x1


class LinearGaussianProbabilityPath(LinearProbabilityPath):

    _require_prng: bool = True

    def __init__(
        self,
        sigma: float,
        prng: torch.Generator | None = None,
    ) -> None:
        super().__init__(sigma=sigma, prng=prng)

    def compute_sigma_t(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ones_like(t) * self._sigma

    def compute_ut(
        self,
        t: torch.Tensor,
        xt: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        # handling shapes
        self._verify_shapes(x0, x1)
        return x1 - x0


class SchrodingerBridgeProbabilityPath(LinearProbabilityPath):

    _require_prng: bool = True

    def __init__(self, sigma: float, prng: torch.Generator | None = None, eps: float = 1e-35) -> None:
        self._eps = eps
        super().__init__(sigma=sigma, prng=prng)

    def compute_sigma_t(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        return self._sigma * torch.sqrt(t * (1 - t))

    def compute_ut(
        self,
        t: torch.Tensor,
        xt: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        # handling shapes
        t = broadcast_to_target_shape(t, x0.shape)
        self._verify_shapes(xt, x1)
        self._verify_shapes(x0, x1)
        self._verify_shapes(t, x1)
        # computing coefficients and noise value
        mu_t = self.compute_mu_t(t, x0, x1)
        return (1 - 2 * t) / (2 * t * (1 - t) + self._eps) * (xt - mu_t) + x1 - x0


class LinearDiracProbabilityPath(LinearProbabilityPath):

    _require_prng: bool = False

    def __init__(
        self,
        sigma: float = 0.0,
        prng: torch.Generator | None = None,
    ) -> None:
        super().__init__(sigma=sigma, prng=prng)

    def compute_sigma_t(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros_like(t)

    def compute_ut(
        self,
        t: torch.Tensor,
        xt: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        # handling shapes
        self._verify_shapes(x0, x1)
        return x1 - x0


class VariancePreservingDiracProbabilityPath(BaseProbabilityPath):
    r"""Trigonometric ("variance preserving") interpolant :math:`\cos(\pi t/2) x_0 + \sin(\pi t/2) x_1`.

    Variance preserving because :math:`\cos^2 + \sin^2 = 1`: for independent :math:`x_0, x_1` of equal
    variance, :math:`\mathrm{Var}(x_t)` is constant in :math:`t`, unlike the linear interpolant whose
    variance dips at :math:`t = 1/2`. Standard form, e.g. Albergo & Vanden-Eijnden, *Stochastic
    Interpolants*.

    Orientation is :math:`x_0 \to x_1` (``mu(0) == x0``, ``mu(1) == x1``), matching every other path in
    this module, ``OTFMObjective`` (which binds ``x0=source``, ``x1=target``) and
    ``integrate_translation`` (which starts at ``y0=source`` and integrates ``t: 0 -> 1``). It previously
    had ``x0`` and ``x1`` swapped, so it transported target -> source while the objective and the solver
    both assumed the opposite -- undetected because nothing in the test suite referenced this class.

    Note this path is CURVED, so the endpoint reparameterisation :math:`\hat{x}_1 = x_t + (1-t) u_t`
    (valid for the linear paths) is biased here and must not be used.
    """

    _require_prng: bool = False

    def __init__(
        self,
        sigma: float = 0.0,
        prng: torch.Generator | None = None,
    ) -> None:
        super().__init__(sigma=sigma, prng=prng)

    def compute_mu_t(
        self,
        t: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        # handling shapes
        t = broadcast_to_target_shape(t, x0.shape)
        self._verify_shapes(x0, x1)
        self._verify_shapes(t, x1)
        return torch.cos(0.5 * PI * t) * x0 + torch.sin(0.5 * PI * t) * x1

    def compute_sigma_t(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros_like(t)

    def compute_ut(
        self,
        t: torch.Tensor,
        xt: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        # handling shapes
        t = broadcast_to_target_shape(t, x0.shape)
        self._verify_shapes(x0, x1)
        self._verify_shapes(t, x1)
        return -0.5 * PI * torch.sin(0.5 * PI * t) * x0 + 0.5 * PI * torch.cos(0.5 * PI * t) * x1
