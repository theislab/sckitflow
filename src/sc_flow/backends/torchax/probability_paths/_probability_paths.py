import abc

import jax
from torchax.interop import jax_view, torch_view
from torchax.tensor import Tensor as TorchaxTensor

import sc_flow.backends.jax.probability_paths as jax_paths

__all__ = [
    "BaseProbabilityPath",
    "LinearProbabilityPath",
    "LinearGaussianProbabilityPath",
    "SchrodingerBridgeProbabilityPath",
    "LinearDiracProbabilityPath",
    "VariancePreservingDiracProbabilityPath",
]


class BaseProbabilityPath:
    """Torchax wrapper base class for JAX probability paths.

    Accepts :class:`TorchaxTensor` inputs and delegates all computation to the
    wrapped JAX probability path. Outputs are returned as :class:`TorchaxTensor`.

    The PRNG key for stochastic paths is a plain :class:`jax.Array` passed
    explicitly to :meth:`compute_xt`, matching the JAX backend's functional
    style and enabling :func:`jax.jit`-compilation of the training loop.
    """

    def __init__(self, jax_path: jax_paths.BaseProbabilityPath) -> None:
        self._jax_path = jax_path

    def compute_mu_t(
        self,
        t: TorchaxTensor,
        x0: TorchaxTensor,
        x1: TorchaxTensor,
    ) -> TorchaxTensor:
        return torch_view(self._jax_path.compute_mu_t(jax_view(t), jax_view(x0), jax_view(x1)))

    def compute_sigma_t(
        self,
        t: TorchaxTensor,
    ) -> TorchaxTensor:
        return torch_view(self._jax_path.compute_sigma_t(jax_view(t)))

    def compute_ut(
        self,
        t: TorchaxTensor,
        xt: TorchaxTensor,
        x0: TorchaxTensor,
        x1: TorchaxTensor,
    ) -> TorchaxTensor:
        return torch_view(self._jax_path.compute_ut(jax_view(t), jax_view(xt), jax_view(x0), jax_view(x1)))

    def compute_xt(
        self,
        t: TorchaxTensor,
        x0: TorchaxTensor,
        x1: TorchaxTensor,
        prng: jax.Array | None = None,
    ) -> TorchaxTensor:
        return torch_view(self._jax_path.compute_xt(jax_view(t), jax_view(x0), jax_view(x1), prng))

    @property
    def is_deterministic(self) -> bool:
        return self._jax_path.is_deterministic


class LinearProbabilityPath(BaseProbabilityPath, abc.ABC):
    """Abstract torchax wrapper base class for JAX linear probability paths.

    Mirrors the abstract :class:`jax_paths.LinearProbabilityPath` in the hierarchy
    so that :class:`LinearGaussianProbabilityPath`, :class:`SchrodingerBridgeProbabilityPath`,
    and :class:`LinearDiracProbabilityPath` share a common base for ``isinstance`` checks.
    """


class LinearGaussianProbabilityPath(LinearProbabilityPath):
    """Torchax wrapper for :class:`jax_paths.LinearGaussianProbabilityPath`."""

    def __init__(self, sigma: float) -> None:
        BaseProbabilityPath.__init__(self, jax_paths.LinearGaussianProbabilityPath(sigma))


class SchrodingerBridgeProbabilityPath(LinearProbabilityPath):
    """Torchax wrapper for :class:`jax_paths.SchrodingerBridgeProbabilityPath`."""

    def __init__(self, sigma: float, eps: float = 1e-35) -> None:
        BaseProbabilityPath.__init__(self, jax_paths.SchrodingerBridgeProbabilityPath(sigma, eps))


class LinearDiracProbabilityPath(LinearProbabilityPath):
    """Torchax wrapper for :class:`jax_paths.LinearDiracProbabilityPath`."""

    def __init__(self, sigma: float = 0.0) -> None:
        BaseProbabilityPath.__init__(self, jax_paths.LinearDiracProbabilityPath(sigma))


class VariancePreservingDiracProbabilityPath(BaseProbabilityPath):
    """Torchax wrapper for :class:`jax_paths.VariancePreservingDiracProbabilityPath`."""

    def __init__(self, sigma: float = 0.0) -> None:
        super().__init__(jax_paths.VariancePreservingDiracProbabilityPath(sigma))
