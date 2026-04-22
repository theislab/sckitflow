from sc_flow.backends.torch.metrics._metrics import EnergyDistance, MaximumMeanDiscrepancy, rbf_kernel_torch

METRICS_REGISTRY = {"e-dist": EnergyDistance, "mmd": MaximumMeanDiscrepancy}

__all__ = ["EnergyDistance", "MaximumMeanDiscrepancy", "rbf_kernel_torch", METRICS_REGISTRY]
