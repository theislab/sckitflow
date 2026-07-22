import abc
from dataclasses import dataclass

__all__ = ["BaseData"]


@dataclass(frozen=True)
class BaseData(abc.ABC):
    ...
