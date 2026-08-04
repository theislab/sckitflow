from typing import TypeVar

__all__ = [
    "DataT",
    "DataTreeT",
    "DistributionT",
    "MatchedDistributionsT",
]


# Shared TypeVars for the sampler / data-tree generics. The abstract base classes
# that used to live here (``Distribution``, ``MatchedDistributions``, ``DataTree``)
# were removed; the concrete classes (``DistributionData``, ``MatchedData``,
# ``MappedTree``) carry all behavior. The container generics ``KeyT``/``ValT`` now
# live inline in ``_mixins.py`` via PEP 695; these remain for the sampler classes.
DataT = TypeVar("DataT")
DataTreeT = TypeVar("DataTreeT")
DistributionT = TypeVar("DistributionT")
MatchedDistributionsT = TypeVar("MatchedDistributionsT")
