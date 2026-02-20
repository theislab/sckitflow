from collections.abc import Callable
from typing import Any

import torch

import sc_flow.methods as basemethods
from sc_flow import _constants, _types
from sc_flow.backends.torch import _utils

__all__ = ["FlowMatching", "OTFlowMatching", "GENOT"]


class FlowMatching(basemethods.FlowMatching):
    """TODO."""

    def step_fn(
        self,
        *args: Any,
        batch: dict[str, _types.TensorLike],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Training step for flow matching.

        Basically, it gets source and target states, plus the condition,
        and then computes the velocity field and the loss.
        """
        source, target = batch[_constants.SOURCE_STATE], batch[_constants.TARGET_STATE]
        if self.generate_from_noise:
            if source is not None:
                raise ValueError("If `generate_from_noise` is `True`, `source` must be `None`.")
            latent = torch.randn_like(target, dtype=target.dtype)
        else:
            latent = source

        batch_size = target.shape[0]
        condition = batch.get("condition")
        t = self.time_sampler((batch_size,), device=target.device)
        xt = self.probability_path.compute_xt(t, latent, target)
        ut = self.probability_path.compute_ut(t, latent, target, xt)
        vt = self.vf(t, xt, condition, source=latent)
        loss = torch.nn.functional.mse_loss(vt, ut)
        return loss, {"loss": loss.detach().cpu().item()}

    def predict(self, *args, **kwargs):
        pass


class OTFlowMatching(FlowMatching, basemethods.OTFlowMatching):
    """TODO."""

    def step_fn(self, rng, batch):
        """Single step function of the solver.

        Parameters
        ----------
        rng
            Random number generator.
        batch
            Data batch with keys ``src_cell_data``, ``tgt_cell_data``, and
            ``condition``.

        Returns
        -------
        Loss value.
        """
        src, tgt = batch["src_cell_data"], batch["tgt_cell_data"]
        tmat = self.match_fn(src, tgt)
        src_ixs, tgt_ixs = _utils.sample_joint(tmat)
        batch["src_cell_data"], batch["tgt_cell_data"] = src[src_ixs], tgt[tgt_ixs]

        return super().step_fn(rng=rng, batch=batch)


class GENOT(basemethods.GENOT):
    """TODO."""

    def __init__(
        self,
        data_match_fn: _types.GENOTDataMatchFn,
        *,
        source_dim: int,
        target_dim: int,
        latent_noise_fn: (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None) = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.data_match_fn = data_match_fn
        self.source_dim = source_dim
        if latent_noise_fn is None:
            latent_noise_fn = torch.randn_like
        self.latent_noise_fn = latent_noise_fn

    def step_fn(self, rng: torch.Tensor, batch: dict[str, torch.Tensor]):
        """Single step fn"""

    def predict(self, *args, **kwargs):
        pass
