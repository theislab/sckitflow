from abc import ABC, abstractmethod

import diffrax as dfx
import jax

from sc_flow.backends.torch._types import RealScalarLike


class AbstractBrownianMotion(ABC):
    def __init__(self):
        self._dfx_bm = None

    @abstractmethod
    def _create_dfx_bm(self) -> dfx.AbstractBrownianPath: ...

    def get_bm(self) -> dfx.AbstractBrownianPath:
        if self._dfx_bm is None:
            self._dfx_bm = self._create_dfx_bm()
        return self._dfx_bm


class VirtualBrownianTree(AbstractBrownianMotion):
    def __init__(
        self,
        t0: RealScalarLike,
        t1: RealScalarLike,
        tol: RealScalarLike,
        shape: tuple[int, ...],
        key: int,
        levy_area: type = dfx.BrownianIncrement,
    ):
        super().__init__()
        self.t0 = t0
        self.t1 = t1
        self.tol = tol
        self.shape = shape
        self.key = key
        self.levy_area = levy_area

    def _create_dfx_bm(self) -> dfx.VirtualBrownianTree:
        return dfx.VirtualBrownianTree(
            t0=self.t0,
            t1=self.t1,
            tol=self.tol,
            shape=self.shape,
            key=jax.random.PRNGKey(self.key) if isinstance(self.key, int) else self.key,
            levy_area=self.levy_area,
        )


class UnsafeBrownianPath(AbstractBrownianMotion):
    def __init__(
        self,
        shape: tuple[int, ...],
        key: int,
        levy_area: type = dfx.BrownianIncrement,
    ):
        super().__init__()
        self.shape = shape
        self.key = key
        self.levy_area = levy_area

    def _create_dfx_bm(self) -> dfx.UnsafeBrownianPath:
        return dfx.UnsafeBrownianPath(
            shape=self.shape,
            key=jax.random.PRNGKey(self.key) if isinstance(self.key, int) else self.key,
            levy_area=self.levy_area,
        )
