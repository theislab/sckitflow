"""Back-compat shim: the neutral torch modules moved to :mod:`scfit.nn`. Import from there in new code."""

from scfit.nn._modules import MLP, AdaLNZero1d, FunctionalModule, Resnet1d

__all__ = ["FunctionalModule", "MLP", "Resnet1d", "AdaLNZero1d"]
