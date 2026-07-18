from sc_flow.backends.torch.metrics._metrics import (
    EnergyDistance,
    MaximumMeanDiscrepancy,
    RSquared,
    rbf_kernel_torch,
)

METRICS_REGISTRY = {"e-dist": EnergyDistance, "mmd": MaximumMeanDiscrepancy, "r_squared": RSquared}

__all__ = ["EnergyDistance", "MaximumMeanDiscrepancy", "RSquared", "rbf_kernel_torch", "METRICS_REGISTRY"]
