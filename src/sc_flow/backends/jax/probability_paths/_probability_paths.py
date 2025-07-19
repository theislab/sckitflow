import abc

import jax
import jax.numpy as jnp

from sc_flow._constants import PI
from sc_flow.backends.jax._types import ArrayLike
from sc_flow.backends.jax._utils import broadcast_to_target_shape


class BaseProbabilityPath(abc.ABC):
    r"""Base Class for Conditional Probability Paths :math: `p_t(\boldsymbol{x}_t | \boldsymbol{x}_0, \boldsymbol{x}_1)`.

    :param _require_prng: Whether a Pseudo-Random Numbers Generator is required for the probability path.
        Pseudo-Random Numbers Generators are required for sampling from non-deterministic probability paths
        with the :method:`compute_xt`. A :class: `ValueError` is thrown otherwise
    :type _require_prng: class: `bool`
    """

    _require_prng: bool

    def __init__(
        self,
        sigma: float,
    ) -> None:
        r"""Initializes the probability path.

        Raises :class: `ValueError` when :attr: `self._require_prng` is `True` and :param: `sigma` is not positive.

        :param sigma: Positive scalar for the noise strength of the probability path.
            This will determine the factor :math: `\sigma` by which the time-dependent standard deviation
            :math: `\sigma_t` of the conditional probability path will be scaled.
            For non-deterministic probability paths, this has to be a positive scalar. A :class: `ValueError` is thrown otherwise.
        :type sigma: class: `float`
        """
        # sanity check for positive values
        if not sigma > 0 and not self.is_deterministic:
            msg = f"Argument sigma should be a positive float for non deterministic probability paths. Found {sigma=}"
            raise ValueError(msg)

        self._sigma = sigma

    def _verify_shapes(
        self,
        input_tensor: ArrayLike,
        target_tensor: ArrayLike,
    ) -> None:
        r"""Verifies that the input and the target arrays have the same shape.

        :param input_tensor: The input array.
        :type input_tensor: class: `ArrayLike`

        :param target_tensor: The target array.
        :type target_tensor: class: `ArrayLike`
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
        t: ArrayLike,
        x0: ArrayLike,
        x1: ArrayLike,
    ) -> ArrayLike:
        r"""Computes the mean :math: `\mu_t(\boldsymbol{x}_0, \boldsymbol{x}_1)` of the probability path.

        :param t: The current time index.
        :type t: class: `ArrayLike`

        :param x0: The source state.
        :type x0: class: `ArrayLike`

        :param x1: The target state.
        :type x1: class: `ArrayLike`
        """

    @abc.abstractmethod
    def compute_sigma_t(
        self,
        t: ArrayLike,
    ) -> ArrayLike:
        r"""Computes the standard deviation :math: `\sigma_t` of the probability path.

        :param t: The current time index.
        :type t: class: `ArrayLike`
        """

    @abc.abstractmethod
    def compute_ut(
        self,
        t: ArrayLike,
        xt: ArrayLike,
        x0: ArrayLike,
        x1: ArrayLike,
    ) -> ArrayLike:
        r"""Computes the conditional velocity field :math: `\u_t(\boldsymbol{x}_t | \boldsymbol{x}_0, \boldsymbol{x}_1)` generating the probability path.

        :param t: The current time index.
        :type t: class: `ArrayLike`

        :param xt: The current sample from the conditional probability path.
        :type xt: class: `ArrayLike`

        :param x0: The source state.
        :type x0: class: `ArrayLike`

        :param x1: The target state.
        :type x1: class: `ArrayLike`
        """

    def compute_xt(
        self,
        t: ArrayLike,
        x0: ArrayLike,
        x1: ArrayLike,
        prng: ArrayLike | None = None,
    ) -> ArrayLike:
        r"""Samples from the conditional probability path :math: `p_t(\boldsymbol{x}_t | \boldsymbol{x}_0, \boldsymbol{x}_1)`.

        Samples from the conditional probability paths are computes as:

            .. math::

                \boldsymbol{x}_t = \mu_t(\boldsymbol{x}_0, \boldsymbol{x}_1) + \sigma_t \boldsymbol{z}, \text{ with }\boldsymbol{z}\sim\mathcal{N}(0_d, \mathbb{I}_d)

        For deterministic probability paths (i.e.: the ones with :attr: `self._require_prng` set to `False`),
        only the mean :math: `\mu_t(\boldsymbol{x}_0, \boldsymbol{x}_1)` will be returned.

        :param t: The current time index.
        :type t: class: `ArrayLike`

        :param x0: The source state.
        :type x0: class: `ArrayLike`

        :param x1: The target state.
        :type x1: class: `ArrayLike`

        :param prng: Pseudo-Random Numbers Generator used to generate random numbers. Only needed for
            non deterministic probability paths. Defaults to `None`.
        :type prng: class: `ArrayLike`
        """
        # handling shapes
        t = broadcast_to_target_shape(t, x0.shape)
        self._verify_shapes(x0, x1)
        self._verify_shapes(t, x1)
        # computing coefficients and noise value
        mu_t = self.compute_mu_t(t, x0, x1)
        # sampling noise
        if self._require_prng:
            if prng is None:
                msg = ""
                raise ValueError(msg)
            sigma_t = self.compute_sigma_t(t)
            noise = jax.random.normal(prng, shape=x0.shape)
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

    This class requires a Pseudo-Random Numbers Generator to be instantiated (i.e.: :attr: `self._require_prng` is `True`).

    :param _require_prng: Whether a Pseudo-Random Numbers Generator is required for the probability path.
        Pseudo-Random Numbers Generators are required for non-deterministic probability paths.
    :type _require_prng: class: `bool`
    """

    _require_prng: bool

    def __init__(
        self,
        sigma: float,
    ) -> None:
        r"""Initializes the probability path.

        :param sigma: Positive scalar for the noise strength of the probability path.
            This will determine the factor :math: `\sigma` by which the time-dependent standard deviation
            :math: `\sigma_t` of the conditional probability path will be scaled.
        :type sigma: class: `float`

        :param prng: Pseudo-Random Numbers Generator used to generate random numbers, defaults to `None`.
        :type prng: class: `None`
        """
        super().__init__(sigma=sigma)

    def compute_mu_t(
        self,
        t: ArrayLike,
        x0: ArrayLike,
        x1: ArrayLike,
    ) -> ArrayLike:
        r"""Computes the mean :math: `\mu_t(\boldsymbol{x}_0, \boldsymbol{x}_1)` of the probability path.

        :param t: The current time index.
        :type t: class: `ArrayLike`

        :param x0: The source state.
        :type x0: class: `ArrayLike`

        :param x1: The target state.
        :type x1: class: `ArrayLike`
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
    ) -> None:
        r"""Initializes the probability path.

        :param sigma: Positive scalar for the noise strength of the probability path.
            This will determine the factor :math: `\sigma` by which the time-dependent standard deviation
            :math: `\sigma_t` of the conditional probability path will be scaled.
        :type sigma: class: `float`

        :param prng: Pseudo-Random Numbers Generator used to generate random numbers, defaults to `None`.
        :type prng: class: `None`
        """
        super().__init__(sigma=sigma)

    def compute_sigma_t(
        self,
        t: ArrayLike,
    ) -> ArrayLike:
        r"""Computes the standard deviation :math: `\sigma_t` of the probability path.

        :param t: The current time index.
        :type t: class: `ArrayLike`
        """
        return jnp.ones_like(t) * self._sigma

    def compute_ut(
        self,
        t: ArrayLike,
        xt: ArrayLike,
        x0: ArrayLike,
        x1: ArrayLike,
    ) -> ArrayLike:
        r"""Computes the conditional velocity field :math: `\u_t(\boldsymbol{x}_t | \boldsymbol{x}_0, \boldsymbol{x}_1)` generating the probability path.

        :param t: The current time index. Such argument will be ignored as the conditional velocity field is constant in time.
        :type t: class: `ArrayLike`

        :param xt: The current sample from the conditional probability path.
            Such argument will be ignored as the conditional velocity field is independent of the current interpolation value :math: `\boldsymbol{x}_t`.
        :type xt: class: `ArrayLike`

        :param x0: The source state.
        :type x0: class: `ArrayLike`

        :param x1: The target state.
        :type x1: class: `ArrayLike`
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

    def __init__(self, sigma: float, eps: float = 1e-35) -> None:
        r"""Initializes the gaussian probability path probability paths.

        :param sigma: The noise value for the flow. This will determine the factor :math: `\sigma` for standard deviation :math: `\sigma_t = \sigma\sqrt{t(1-t)}` of the conditional probability path.
        :type sigma: class: `float`

        :param prng: Pseudo-Random Numbers Generator used to generate random numbers.
        :type prng: class: `None`

        :param eps: Small constant to be added to the denominator of the conditional velocity field for numerical stability, defaults to :math: `10^{-35}`.
        :type eps: class: `float`
        """
        self._eps = eps
        super().__init__(sigma=sigma)

    def compute_sigma_t(
        self,
        t: ArrayLike,
    ) -> ArrayLike:
        r"""Computes the standard deviation :math: `\sigma_t` of the probability path.

        :param t: The current time index.
        :type t: class: `ArrayLike`
        """
        return self._sigma * jnp.sqrt(t * (1 - t))

    def compute_ut(
        self,
        t: ArrayLike,
        xt: ArrayLike,
        x0: ArrayLike,
        x1: ArrayLike,
    ) -> ArrayLike:
        r"""Computes the conditional velocity field :math: `\u_t(\boldsymbol{x}_t | \boldsymbol{x}_0, \boldsymbol{x}_1)` generating the probability path.

        :param t: The current time index. Such argument will be ignored as the conditional velocity field is constant in time.
        :type t: class: `ArrayLike`

        :param xt: The current sample from the conditional probability path.
            Such argument will be ignored as the conditional velocity field is independent of the current interpolation value :math: `\boldsymbol{x}_t`.
        :type xt: class: `ArrayLike`

        :param x0: The source state.
        :type x0: class: `ArrayLike`

        :param x1: The target state.
        :type x1: class: `ArrayLike`
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
    ) -> None:
        super().__init__(sigma=sigma)

    def compute_sigma_t(
        self,
        t: ArrayLike,
    ) -> ArrayLike:
        r"""Computes the standard deviation :math: `\sigma_t` of the probability path.

        This method always returns zero.

        :param t: The current time index at which to compute the interpolation.
        :type t: class: `ArrayLike`
        """
        return jnp.zeros_like(t)

    def compute_ut(
        self,
        t: ArrayLike,
        xt: ArrayLike,
        x0: ArrayLike,
        x1: ArrayLike,
    ) -> ArrayLike:
        r"""Computes the conditional velocity field :math: `\u_t(\boldsymbol{x}_t | \boldsymbol{x}_0, \boldsymbol{x}_1)` generating the probability path.

        :param t: The current time index at which to compute the interpolation.
            Such argument will be ignored as the conditional velocity field is constant in time.
        :type t: class: `ArrayLike`

        :param xt: The current sample from the conditional probability path.
            Such argument will be ignored as the conditional velocity field is independent of the current interpolation value :math: `\boldsymbol{x}_t`.
        :type xt: class: `ArrayLike`

        :param x0: The source state.
        :type x0: class: `ArrayLike`

        :param x1: The target state.
        :type x1: class: `ArrayLike`
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
    ) -> None:
        super().__init__(sigma=sigma)

    def compute_mu_t(
        self,
        t: ArrayLike,
        x0: ArrayLike,
        x1: ArrayLike,
    ) -> ArrayLike:
        r"""Computes the mean :math: `\mu_t(\boldsymbol{x}_0, \boldsymbol{x}_1)` of the probability path.

        :param t: The current time index.
        :type t: class: `ArrayLike`

        :param x0: The source state.
        :type x0: class: `ArrayLike`

        :param x1: The target state.
        :type x1: class: `ArrayLike`
        """
        # handling shapes
        t = broadcast_to_target_shape(t, x0.shape)
        self._verify_shapes(x0, x1)
        self._verify_shapes(t, x1)
        return jnp.cos(0.5 * PI * t) * x1 + jnp.sin(0.5 * PI * t) * x0

    def compute_sigma_t(
        self,
        t: ArrayLike,
    ) -> ArrayLike:
        r"""Computes the standard deviation :math: `\sigma_t` of the probability path.

        This method always returns zero.

        :param t: The current time index at which to compute the interpolation.
        :type t: class: `ArrayLike`
        """
        return jnp.zeros_like(t)

    def compute_ut(
        self,
        t: ArrayLike,
        xt: ArrayLike,
        x0: ArrayLike,
        x1: ArrayLike,
    ) -> ArrayLike:
        r"""Computes the conditional velocity field :math: `\u_t(\boldsymbol{x}_t | \boldsymbol{x}_0, \boldsymbol{x}_1)` generating the probability path.

        :param t: The current time index at which to compute the interpolation.
            Such argument will be ignored as the conditional velocity field is constant in time.
        :type t: class: `ArrayLike`

        :param xt: The current sample from the conditional probability path.
            Such argument will be ignored as the conditional velocity field is independent of the current interpolation value :math: `\boldsymbol{x}_t`.
        :type xt: class: `ArrayLike`

        :param x0: The source state.
        :type x0: class: `ArrayLike`

        :param x1: The target state.
        :type x1: class: `ArrayLike`
        """
        # handling shapes
        t = broadcast_to_target_shape(t, x0.shape)
        self._verify_shapes(x0, x1)
        self._verify_shapes(t, x1)
        return -0.5 * PI * jnp.sin(0.5 * PI * t) * x1 + 0.5 * PI * jnp.cos(0.5 * PI * t) * x0
