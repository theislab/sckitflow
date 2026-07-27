"""Back-compat shim: the model-family registry moved to :mod:`scfit.families`. Import from there in new code."""

from scfit.families import (
    ENTRY_POINT_GROUP,
    FAMILY_REGISTRY,
    FamilyBuilder,
    FamilyFactory,
    available_families,
    build_family,
    load_family,
    register_family,
)

__all__ = [
    "FamilyBuilder",
    "FamilyFactory",
    "FAMILY_REGISTRY",
    "register_family",
    "available_families",
    "load_family",
    "build_family",
]
