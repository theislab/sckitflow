import abc
import logging

import torch

from sc_flow._constants import PI
from sc_flow.backends.torch._utils import broadcast_to_target_shape

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
    r"""Base Class for Conditional Probability Paths :math: `p_t(\boldsymbol{x}_t | \boldsymbol{x}_0, \boldsymbol{x}_1)`.

    :param _require_prng: Whether a Pseudo-Random Numbers Generator is required for the probability path.
        Pseudo-Random Numbers Generators are required for reproducibility of non-deterministic probability paths and should be instances of
        :class: `torch.Generator`, when provided. For non-deterministic probability paths a warning is displayed and it is set to the
        output of :constant: `torch.random.default_generator`.
    :type _require_prng: class: `bool`
    """

    _require_prng: bool

    def __init__(
        self,
        sigma: float,
        prng: torch.Generator | None = None,
    ) -> None:
        r"""Initializes the probability path.

        :param sigma: Positive scalar for the noise strength of the probability path.
            This will determine the factor :math: `\sigma` by which the time-dependent standard deviation
            :math: `\sigma_t` of the conditional probability path will be scaled.
            For non-deterministic probability paths, this has to be a positive scalar. A :class: `ValueError` is thrown otherwise.
        :type sigma: class: `float`

        :param prng: Pseudo-Random Numbers Generator used to generate random numbers. Only needed for
            non deterministic probability paths. Defaults to `None`, in which case it is set to the
            output of :constant: `torch.random.default_generator`.
        :type prng: class: `torch.Generator | None`
        """
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
        r"""Verifies that the input and the target tensors have the same shape.

        :param input_tensor: The input tensor.
        :type input_tensor: class: `torch.Tensor`

        :param target_tensor: The target tensor.
        :type target_tensor: class: `torch.Tensor`
        """
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
        r"""Computes the mean :math: `\mu_t(\boldsymbol{x}_0, \boldsymbol{x}_1)` of the probability path.

        :param t: The current time index.
        :type t: class: `torch.Tensor`

        :param x0: The source state.
        :type x0: class: `torch.Tensor`

        :param x1: The target state.
        :type x1: class: `torch.Tensor`
        """

    @abc.abstractmethod
    def compute_sigma_t(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        r"""Computes the standard deviation :math: `\sigma_t` of the probability path.

        :param t: The current time index.
        :type t: class: `torch.Tensor`
        """

    @abc.abstractmethod
    def compute_ut(
        self,
        t: torch.Tensor,
        xt: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        r"""Computes the conditional velocity field :math: `\u_t(\boldsymbol{x}_t | \boldsymbol{x}_0, \boldsymbol{x}_1)` generating the probability path.

        :param t: The current time index.
        :type t: class: `torch.Tensor`

        :param xt: The current sample from the conditional probability path.
        :type xt: class: `torch.Tensor`

        :param x0: The source state.
        :type x0: class: `torch.Tensor`

        :param x1: The target state.
        :type x1: class: `torch.Tensor`
        """

    def compute_xt(
        self,
        t: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        r"""Samples from the conditional probability path :math: `p_t(\boldsymbol{x}_t | \boldsymbol{x}_0, \boldsymbol{x}_1)`.

        Samples from the conditional probability paths are computes as:

            .. math::

                \boldsymbol{x}_t = \mu_t(\boldsymbol{x}_0, \boldsymbol{x}_1) + \sigma_t \boldsymbol{z}, \text{ with }\boldsymbol{z}\sim\mathcal{N}(0_d, \mathbb{I}_d)

        For deterministic probability paths (i.e.: the ones with :attr: `self._require_prng` set to `False`),
        only the mean :math: `\mu_t(\boldsymbol{x}_0, \boldsymbol{x}_1)` will be returned.

        :param t: The current time index.
        :type t: class: `torch.Tensor`

        :param x0: The source state.
        :type x0: class: `torch.Tensor`

        :param x1: The target state.
        :type x1: class: `torch.Tensor`
        """
        # handling shapes
        t = broadcast_to_target_shape(t, x0.shape)
        self._verify_shapes(x0, x1)
        self._verify_shapes(t, x1)
        # computing coefficients and noise value
        mu_t = self.compute_mu_t(t, x0, x1)
        # sampling noise
        if self._require_prng:
            sigma_t = self.compute_sigma_t(t)
            noise = torch.randn(*x0.shape, generator=self._prng, device=x0.device)
            return mu_t + sigma_t * noise
        # returning mean for deterministic paths
        return mu_t

    @classmethod
    @property
    def is_deterministic(
        cls,
    ) -> bool:
        """Flag indicating whether the probability path is deterministic.

        As non-deterministic probability paths require a Pseudo-Random Numbers Generator,
        this will return `False` when such generator is not required.
        """
        return not cls._require_prng


class LinearProbabilityPath(BaseProbabilityPath, abc.ABC):
    r"""Class implementing Probability Paths with interpolation linear in time.

    Where the mean \mu_t(\boldsymbol{x}_0, \boldsymbol{x}_1) is computed as:

        .. math::
            \mu_t(\boldsymbol{x}_0, \boldsymbol{x}_1) = (1 - t)\boldsymbol{x}_0 + t\boldsymbol{x}_1

    :param _require_prng: Whether a Pseudo-Random Numbers Generator is required for the probability path.
        Pseudo-Random Numbers Generators are required for non-deterministic probability paths.
    :type _require_prng: class: `bool`
    """

    _require_prng: bool

    def __init__(
        self,
        sigma: float,
        prng: torch.Generator | None = None,
    ) -> None:
        r"""Initializes the probability path.

        :param sigma: Positive scalar for the noise strength of the probability path.
            This will determine the factor :math: `\sigma` by which the time-dependent standard deviation
            :math: `\sigma_t` of the conditional probability path will be scaled.
        :type sigma: class: `float`

        :param prng: Pseudo-Random Numbers Generator used to generate random numbers, defaults to `None`.
        :type prng: class: `None`
        """
        super().__init__(sigma=sigma, prng=prng)

    def compute_dot_alpha_t(self, t):
        return torch.ones_like(t)

    def compute_alpha_t(self, t):
        return t

    @abc.abstractmethod
    def compute_dot_sigma_t(self, t): ...

    def compute_mu_t(
        self,
        t: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        r"""Computes the mean :math: `\mu_t(\boldsymbol{x}_0, \boldsymbol{x}_1)` of the probability path.

        :param t: The current time index.
        :type t: class: `torch.Tensor`

        :param x0: The source state.
        :type x0: class: `torch.Tensor`

        :param x1: The target state.
        :type x1: class: `torch.Tensor`
        """
        # handling shapes
        t = broadcast_to_target_shape(t, x0.shape)
        self._verify_shapes(x0, x1)
        self._verify_shapes(t, x1)

        return (1 - t) * x0 + t * x1


class LinearGaussianProbabilityPath(LinearProbabilityPath):
    r"""Class implementing Constant Noise Gaussian Probability Paths with linear interpolation.

    The Constant Noise Gaussian Probability Path is defined as:

        .. math::
            p_t(\boldsymbol{x}_t | \boldsymbol{x}_0, \boldsymbol{x}_1) = \mathcal{N}(\boldsymbol{x}_t | \mu_t(\boldsymbol{x}_0, \boldsymbol{x}_1), \sigma^2\mathbb{I}_d)


    Where the mean \mu_t(\boldsymbol{x}_0, \boldsymbol{x}_1) is computed as:

        .. math::
            \mu_t(\boldsymbol{x}_0, \boldsymbol{x}_1) = (1 - t)\boldsymbol{x}_0 + t\boldsymbol{x}_1

    The Conditional Velocity Field Generating the Constant Noise Gaussian Probability Path is defined as:

        .. math::
            u_t(\boldsymbol{x}_t | \boldsymbol{x}_0, \boldsymbol{x}_1) = \boldsymbol{x}_1 - \boldsymbol{x}_0

    This class requires a Pseudo-Random Numbers Generator to be instantiated (i.e.: :attr: `self._require_prng` is `True`).
    """

    _require_prng: bool = True

    def __init__(
        self,
        sigma: float,
        prng: torch.Generator | None = None,
    ) -> None:
        r"""Initializes the probability path.

        :param sigma: Positive scalar for the noise strength of the probability path.
            This will determine the factor :math: `\sigma` by which the time-dependent standard deviation
            :math: `\sigma_t` of the conditional probability path will be scaled.
        :type sigma: class: `float`

        :param prng: Pseudo-Random Numbers Generator used to generate random numbers.
        :type prng: class: `None`
        """
        super().__init__(sigma=sigma, prng=prng)

    def compute_sigma_t(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        r"""Computes the standard deviation :math: `\sigma_t` of the probability path.

        :param t: The current time index.
        :type t: class: `torch.Tensor`
        """
        return torch.ones_like(t) * self._sigma

    def compute_dot_sigma_t(self, t):
        return torch.zeros_like(t)

    def compute_ut(
        self,
        t: torch.Tensor,
        xt: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        r"""Computes the conditional velocity field :math: `\u_t(\boldsymbol{x}_t | \boldsymbol{x}_0, \boldsymbol{x}_1)` generating the probability path.

        :param t: The current time index. Such argument will be ignored as the conditional velocity field is constant in time.
        :type t: class: `torch.Tensor`

        :param xt: The current sample from the conditional probability path.
            Such argument will be ignored as the conditional velocity field is independent of the current interpolation value :math: `\boldsymbol{x}_t`.
        :type xt: class: `torch.Tensor`

        :param x0: The source state.
        :type x0: class: `torch.Tensor`

        :param x1: The target state.
        :type x1: class: `torch.Tensor`
        """
        # handling shapes
        self._verify_shapes(x0, x1)
        return x1 - x0


class SchrodingerBridgeProbabilityPath(LinearProbabilityPath):
    r"""Class implementing Schrodinger Bridge Probability Paths with linear interpolation.

    Subclasses :class: `LinearGaussianProbabilityPath` and overrides its :method: `.compute_sigma_t` and :method: `.compute_ut` methods.

    The  Schrodinger Bridge  Probability Path is defined as:

        .. math::
            p_t(\boldsymbol{x}_t | \boldsymbol{x}_0, \boldsymbol{x}_1) = \mathcal{N}(\boldsymbol{x}_t | \mu_t(\boldsymbol{x}_0, \boldsymbol{x}_1), \sigma_t^2\mathbb{I}_d)

    Where the mean \mu_t(\boldsymbol{x}_0, \boldsymbol{x}_1) is computed as:

        .. math::
            \mu_t(\boldsymbol{x}_0, \boldsymbol{x}_1) = (1 - t)\boldsymbol{x}_0 + t\boldsymbol{x}_1

    And the standard deviation reads:
        ..math::
            \sigma_t = \sqrt{t(1-t)}\sigma

    The Conditional Velocity Field Generating the Schrodinger Bridge Probability Path is defined as:

        .. math::
            u_t(\boldsymbol{x}_t | \boldsymbol{x}_0, \boldsymbol{x}_1) = \frac{1 - 2t}{2t(1 - t)}(\boldsymbol{x}_t - \mu_t(\boldsymbol{x}_0, \boldsymbol{x}_1))+ (\boldsymbol{x}_1 - \boldsymbol{x}_0)

    For numerical stability in the computation of :math: `u_t(\boldsymbol{x}_t | \boldsymbol{x}_0, \boldsymbol{x}_1)` at times :math: `t=0` and :math: `t=1` a small scalar is added to the denominator.

    This class requires a Pseudo-Random Numbers Generator to be instantiated (i.e.: :attr: `self._require_prng` is `True`).
    """

    _require_prng: bool = True

    def __init__(
        self, sigma: float, sigma_min: float = 0.0, prng: torch.Generator | None = None, eps: float = 1e-35
    ) -> None:
        r"""Initializes the gaussian probability path probability paths.

        :param sigma: The noise value for the flow. This will determine the factor :math: `\sigma` for standard deviation :math: `\sigma_t = \sigma\sqrt{t(1-t)}` of the conditional probability path.
        :type sigma: class: `float`

        :param prng: Pseudo-Random Numbers Generator used to generate random numbers.
        :type prng: class: `None`

        :param eps: Small constant to be added to the denominator of the conditional velocity field for numerical stability, defaults to :math: `10^{-35}`.
        :type eps: class: `float`
        """
        self._eps = eps
        self._sigma_min = sigma_min
        super().__init__(sigma=sigma, prng=prng)

    def compute_sigma_t(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        r"""Computes the standard deviation :math: `\sigma_t` of the probability path.

        :param t: The current time index.
        :type t: class: `torch.Tensor`
        """
        return self._sigma * torch.sqrt(t * (1 - t))

    def compute_dot_sigma_t(self, t):
        den = torch.sqrt(t * (1 - t))
        num = 1 - 2 * t
        return 0.5 * self._sigma * num / den

    def compute_ut(
        self,
        t: torch.Tensor,
        xt: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        r"""Computes the conditional velocity field :math: `\u_t(\boldsymbol{x}_t | \boldsymbol{x}_0, \boldsymbol{x}_1)` generating the probability path.

        :param t: The current time index. Such argument will be ignored as the conditional velocity field is constant in time.
        :type t: class: `torch.Tensor`

        :param xt: The current sample from the conditional probability path.
            Such argument will be ignored as the conditional velocity field is independent of the current interpolation value :math: `\boldsymbol{x}_t`.
        :type xt: class: `torch.Tensor`

        :param x0: The source state.
        :type x0: class: `torch.Tensor`

        :param x1: The target state.
        :type x1: class: `torch.Tensor`
        """
        # handling shapes
        t = broadcast_to_target_shape(t, x0.shape)
        self._verify_shapes(xt, x1)
        self._verify_shapes(x0, x1)
        self._verify_shapes(t, x1)
        # computing coefficients and noise value
        mu_t = self.compute_mu_t(t, x0, x1)
        return (1 - 2 * t) / (2 * t * (1 - t) + self._eps) * (xt - mu_t) + x1 - x0


class LinearDiracProbabilityPath(LinearProbabilityPath):
    r"""Class implementing Deterministic Dirac Probability Paths.

    The Deterministic Dirac Probability Paths is defined as:

        .. math::
            p_t(\boldsymbol{x}_t | \boldsymbol{x}_0, \boldsymbol{x}_1) = \delta(\boldsymbol{x}_t  - \mu_t(\boldsymbol{x}_0, \boldsymbol{x}_1))

    Where :math: `\delta(\cdot)` is the Dirac point measure centered around 0.

    The Mean of the Dirac Probability Path is defined as:

        .. math::
            \mu_t(\boldsymbol{x}_0, \boldsymbol{x}_1) = (1 - t)\boldsymbol{x}_0 + t\boldsymbol{x}_1

    The Conditional Velocity Field Generating the Dirac Probability Path is defined as:

        .. math::
            u_t(\boldsymbol{x}_t | \boldsymbol{x}_0, \boldsymbol{x}_1) = \boldsymbol{x}_1 - \boldsymbol{x}_0

    It is deterministic and it does not require a Pseudo-Random Numbers Generator to be instantiated.
    """

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
        r"""Computes the standard deviation :math: `\sigma_t` of the probability path.

        This method always returns zero.

        :param t: The current time index at which to compute the interpolation.
        :type t: class: `torch.Tensor`
        """
        return torch.zeros_like(t)

    def compute_dot_sigma_t(self, t):
        return torch.zeros_like(t)

    def compute_ut(
        self,
        t: torch.Tensor,
        xt: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        r"""Computes the conditional velocity field :math: `\u_t(\boldsymbol{x}_t | \boldsymbol{x}_0, \boldsymbol{x}_1)` generating the probability path.

        :param t: The current time index at which to compute the interpolation.
            Such argument will be ignored as the conditional velocity field is constant in time.
        :type t: class: `torch.Tensor`

        :param xt: The current sample from the conditional probability path.
            Such argument will be ignored as the conditional velocity field is independent of the current interpolation value :math: `\boldsymbol{x}_t`.
        :type xt: class: `torch.Tensor`

        :param x0: The source state.
        :type x0: class: `torch.Tensor`

        :param x1: The target state.
        :type x1: class: `torch.Tensor`
        """
        # handling shapes
        self._verify_shapes(x0, x1)
        return x1 - x0


class VariancePreservingDiracProbabilityPath(BaseProbabilityPath):
    r"""Class implementing Variance Preserving Probability Paths.

    The Deterministic Variance Preserving Probability Path is defined as:

        .. math::
            p_t(\boldsymbol{x}_t | \boldsymbol{x}_0, \boldsymbol{x}_1) = \delta(\boldsymbol{x}_t  - \mu_t(\boldsymbol{x}_0, \boldsymbol{x}_1))

    Where :math: `\delta(\cdot)` is the Dirac point measure centered at 0.

    The Mean of the Variance Preserving Probability Path is defined as:

        .. math::
            \mu_t(\boldsymbol{x}_0, \boldsymbol{x}_1) = \sin(\frac{1}{2}\pi t)\boldsymbol{x}_0 + \cos(\frac{1}{2}\pi t)\boldsymbol{x}_1

    The Conditional Velocity Field Generating the Variance Preserving Probability Path is defined as:

        .. math::
            u_t(\boldsymbol{x}_t | \boldsymbol{x}_0, \boldsymbol{x}_1) = \frac{1}{2}\pi \cos(\frac{1}{2}\pi t)\boldsymbol{x}_0 - \frac{1}{2}\pi\cos(\frac{1}{2}\pi t)\boldsymbol{x}_1

    It is deterministic and it does not require a Pseudo-Random Numbers Generator to be instantiated.
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
        r"""Computes the mean :math: `\mu_t(\boldsymbol{x}_0, \boldsymbol{x}_1)` of the probability path.

        :param t: The current time index.
        :type t: class: `torch.Tensor`

        :param x0: The source state.
        :type x0: class: `torch.Tensor`

        :param x1: The target state.
        :type x1: class: `torch.Tensor`
        """
        # handling shapes
        t = broadcast_to_target_shape(t, x0.shape)
        self._verify_shapes(x0, x1)
        self._verify_shapes(t, x1)
        return torch.cos(0.5 * PI * t) * x0 + torch.sin(0.5 * PI * t) * x1

    def compute_sigma_t(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        r"""Computes the standard deviation :math: `\sigma_t` of the probability path.

        This method always returns zero.

        :param t: The current time index at which to compute the interpolation.
        :type t: class: `torch.Tensor`
        """
        return self._sigma * torch.zeros_like(t)

    def compute_ut(
        self,
        t: torch.Tensor,
        xt: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        r"""Computes the conditional velocity field :math: `\u_t(\boldsymbol{x}_t | \boldsymbol{x}_0, \boldsymbol{x}_1)` generating the probability path.

        :param t: The current time index at which to compute the interpolation.
            Such argument will be ignored as the conditional velocity field is constant in time.
        :type t: class: `torch.Tensor`

        :param xt: The current sample from the conditional probability path.
            Such argument will be ignored as the conditional velocity field is independent of the current interpolation value :math: `\boldsymbol{x}_t`.
        :type xt: class: `torch.Tensor`

        :param x0: The source state.
        :type x0: class: `torch.Tensor`

        :param x1: The target state.
        :type x1: class: `torch.Tensor`
        """
        # handling shapes
        t = broadcast_to_target_shape(t, x0.shape)
        self._verify_shapes(x0, x1)
        self._verify_shapes(t, x1)
        return -0.5 * PI * torch.sin(0.5 * PI * t) * x0 + 0.5 * PI * torch.cos(0.5 * PI * t) * x1
