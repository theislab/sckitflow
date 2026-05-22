import abc

import numpy as np

from sc_flow.data.containers._distribution import DistributionData
from sc_flow.external._context import ExternalModelContext
from sc_flow.preprocessing.transforms._base import BaseTransform

__all__ = ["BasePreprocessing"]


class BasePreprocessing(abc.ABC):
    """Base class for preprocessing distribution data.

    Derived classes should define the `_extract_data`
    and `_write_data` methods and should be designed
    to operate on :class: `DistributionData` objects.
    """

    def __init__(
        self,
        transform: BaseTransform | None = None,
        encoder_context: ExternalModelContext | None = None,
        decoder_context: ExternalModelContext | None = None,
    ) -> None:
        """Initializes the preprocessing module.

        :param transform: The transformation to be applied to the data.
            Defaults to `None`.
        :type transform: class: `BaseTransform | None`

        :param encoder_context: The context for optional encoder models.
            Defaults to `None`.
        :type encoder_context: class: `ExternalModelContext | None`

        :param decoder_context: The context for optional encoder models.
            Defaults to `None`.
        :type decoder_context: class: `ExternalModelContext | None`
        """
        self._transform = transform
        self._encoder_context = encoder_context
        self._decoder_context = decoder_context

        self._is_fitted: bool = False

    @abc.abstractmethod
    def _extract_data(self, data: DistributionData) -> np.ndarray: ...

    @abc.abstractmethod
    def _write_data(self, X: np.ndarray, data: DistributionData) -> DistributionData: ...

    def _fit_transform(self, X: np.ndarray) -> None:
        if self._transform is not None:
            self._transform.fit(X)

    def _apply_transform(self, X: np.ndarray) -> np.ndarray:
        if self._transform is not None:
            X = self._transform.transform(X)
        if self._encoder_context is not None:
            X = self._encoder_context(X)
        return X

    def _apply_inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self._decoder_context is not None:
            X = self._decoder_context(X)
        if self._transform is not None:
            X = self._transform.inverse_transform(X)
        return X

    def fit(self, data: DistributionData) -> None:
        """Fits the transformation on the input distribution.

        :param data: The input distribution data to fit the transform on.
        :type data: class: `DistributionData`
        """
        X = self._extract_data(data)
        self._fit_transform(X)
        self._is_fitted = True

    def transform(self, data: DistributionData) -> DistributionData:
        """Applies the transformation to the input distribution.

        The :method:`self.fit` method should be called before
        transforming the data. Raises a :class: `RuntimeError`
        otherwise.

        :param data: The input distribution data to transform.
        :type data: class: `DistributionData`
        """
        # transform need to be fitted
        if not self.is_fitted:
            raise RuntimeError("Preprocessing not fitted. Call fit() first.")

        # apply transformation
        X = self._extract_data(data)
        X = self._apply_transform(X)
        return self._write_data(X, data)

    def inverse_transform(self, data: DistributionData) -> DistributionData:
        """Applies the inverse transformation to the input distribution.

        The :method:`self.fit` method should be called before
        transforming the data. Raises a :class: `RuntimeError`
        otherwise.

        :param data: The input distribution data which
            to apply the inverse transformation on.
        :type data: class: `DistributionData`
        """
        # transform need to be fitted
        if not self.is_fitted:
            raise RuntimeError("Preprocessing not fitted. Call fit() first.")

        # apply inverse transformation
        X = self._extract_data(data)
        X = self._apply_inverse_transform(X)
        return self._write_data(X, data)

    @property
    def is_fitted(self) -> bool:
        """Whether the preprocessing module was fitted."""
        if self._transform is None:
            return self._is_fitted
        return self._transform.is_fitted and self._is_fitted
