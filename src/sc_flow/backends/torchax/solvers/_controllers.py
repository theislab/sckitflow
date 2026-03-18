from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import diffrax as dfx
from jaxtyping import PyTree

from sc_flow.backends.torch._types import RealScalarLike


class AbstractStepSizeController(ABC):
    def __init__(self):
        self._dfx_controller = None

    @abstractmethod
    def _create_dfx_controller(self): ...
    def get_controller(self):
        if self._dfx_controller is None:
            self._dfx_controller = self._create_dfx_controller()
        return self._dfx_controller


class ConstantStepSizeController(AbstractStepSizeController):
    def __init__(self):
        super().__init__()

    def _create_dfx_controller(self):
        return dfx.ConstantStepSize()


class PIDController(AbstractStepSizeController):
    def __init__(
        self,
        rtol: RealScalarLike = 1e-3,
        atol: RealScalarLike = 1e-6,
        norm: Callable[[PyTree], RealScalarLike] = None,
        pcoeff: RealScalarLike = 0,
        icoeff: RealScalarLike = 1,
        dcoeff: RealScalarLike = 0,
        dtmin: RealScalarLike | None = None,
        dtmax: RealScalarLike | None = None,
        force_dtmin: bool = True,
        factormin: RealScalarLike = 0.2,
        factormax: RealScalarLike = 10.0,
        safety: RealScalarLike = 0.9,
        error_order: RealScalarLike | None = None,
    ):
        super().__init__()
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.pcoeff = pcoeff
        self.icoeff = icoeff
        self.dcoeff = dcoeff
        self.dtmin = dtmin
        self.dtmax = dtmax
        self.force_dtmin = force_dtmin
        self.factormin = factormin
        self.factormax = factormax
        self.safety = safety
        self.error_order = error_order

    def _create_dfx_controller(self):
        kwargs = {
            "rtol": self.rtol,
            "atol": self.atol,
            "pcoeff": self.pcoeff,
            "icoeff": self.icoeff,
            "dcoeff": self.dcoeff,
            "dtmin": self.dtmin,
            "dtmax": self.dtmax,
            "force_dtmin": self.force_dtmin,
            "factormin": self.factormin,
            "factormax": self.factormax,
            "safety": self.safety,
            "error_order": self.error_order,
        }

        if self.norm is not None:
            kwargs["norm"] = self.norm

        return dfx.PIDController(**kwargs)


class StepTo(AbstractStepSizeController):
    def __init__(self, ts: Sequence[float]):
        super().__init__()
        if (ts.ndim != 1) or (len(ts) < 2):
            raise ValueError("ts must be a 1D sequence with at least two time points.")
        self.ts = ts

    def _create_dfx_controller(self):
        return dfx.StepTo(ts=self.ts)


class ClipStepSizeController(AbstractStepSizeController):
    def __init__(
        self,
        controller: AbstractStepSizeController,
        step_ts: Sequence[float] | None = None,
        jump_ts: Sequence[float] | None = None,
        store_rejected_steps: int | None = None,
    ):
        super().__init__()
        self.controller = controller
        self.step_ts = step_ts
        self.jump_ts = jump_ts
        self.store_rejected_steps = store_rejected_steps

    def _create_dfx_controller(self):
        return dfx.ClipStepSizeController(
            controller=self.controller.get_controller(),
            step_ts=self.step_ts,
            jump_ts=self.jump_ts,
            store_rejected_steps=self.store_rejected_steps,
        )
