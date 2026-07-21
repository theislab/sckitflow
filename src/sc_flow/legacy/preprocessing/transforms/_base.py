import abc
from collections.abc import Collection
from dataclasses import dataclass

import numpy as np

__all__ = ["BaseTransform", "TransformParams"]


@dataclass
class TransformParams:
    """Base class for storing fitted preprocessor parameters.

    This is a container for parameters learned during the fit process.
    Each specific preprocessor will define its own parameter class.

    Attributes
    ----------
        is_fitted: Whether the preprocessor has been fitted to data.
        n_features: Number of features in the fitted data.
    """

    is_fitted: bool = False
    n_features: int | None = None

    def _validate_fitted(self) -> None:
        """Check that the preprocessor has been fitted.

        Raises
        ------
            RuntimeError: If the preprocessor has not been fitted yet.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "This preprocessor has not been fitted yet. "
                "Call 'fit' or 'fit_transform' before using 'transform' or 'inverse_transform'."
            )

    def _validate_n_features(self, X: np.ndarray) -> None:
        """Check that input has correct number of features.

        Args:
            X: Input array to validate, shape (n_samples, n_features).

        Raises
        ------
            ValueError: If number of features doesn't match fitted data.
        """
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Input has {X.shape[1]} features, but preprocessor was fitted with {self.n_features} features."
            )


class BaseTransform(abc.ABC):
    """Abstract base class for all preprocessors.

    Preprocessors work with class:'numpy' objects of shape (n_samples, n_features).

    The derived classes will need to override the following abstract methods to be instantiated:
        - _fit: Learn parameters from data
        - _transform: Apply transformation using learned parameters
        - _inverse_transform: Reverse the transformation
    Maybe define  an immutable data structure for the returned object?
    """

    def __init__(self) -> None:
        """Initialize the preprocessor.

        The params attribute will store fitted parameters.
        It should be set to a specific TransformParams subclass in subclasses.
        """
        self._params: TransformParams | Collection[TransformParams] | None = None

    @property
    def params(self) -> TransformParams | Collection[TransformParams]:
        """Get the fitted parameters.

        Returns
        -------
            The fitted parameters object.

        Raises
        ------
            RuntimeError: If the preprocessor has not been fitted yet.
        """
        if self._params is None:
            raise RuntimeError("Preprocessor has not been fitted yet. Call 'fit()' or 'fit_transform()' first.")
        return self._params

    @property
    def is_fitted(self) -> bool:
        """Check if the preprocessor has been fitted.

        Returns
        -------
            True if fitted, False otherwise.
        """
        return self._params is not None and self._params.is_fitted

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit to data, then transform it.

        This is a convenience method that combines fit and transform.

        Args:
            X: Input data array, shape (n_samples, n_features).

        Returns
        -------
            Transformed data array, shape (n_samples, n_features_out).
        """
        self.fit(X)
        return self.transform(X)

    def _validate_input(self, X: np.ndarray, name: str = "X") -> None:
        """Validate input array shape and type.

        Args:
            X: Input array to validate.
            name: Name of the parameter for error messages.

        Raises
        ------
            TypeError: If X is not a numpy array.
            ValueError: If X is not 2D or has invalid shape.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError(f"{name} must be a numpy array, got {type(X)}")

        if X.ndim != 2:
            raise ValueError(f"{name} must be 2D array with shape (n_samples, n_features), got shape {X.shape}")

        if X.shape[0] == 0:
            raise ValueError(f"{name} must have at least one sample")

        if X.shape[1] == 0:
            raise ValueError(f"{name} must have at least one feature")

    @abc.abstractmethod
    def fit(self, X: np.ndarray) -> "BaseTransform":
        """Fit the preprocessor to the data.

        This method learns parameters from the input data and stores them
        in self._params. It should:
        1. Validate input shape
        2. Compute statistics/parameters
        3. Store parameters in a _TransformParams object
        4. Set is_fitted=True
        5. Return self for method chaining

        Args:
            X: Input data array, shape (n_samples, n_features).

        Returns
        -------
            Self

        Raises
        ------
            ValueError: If input array has invalid shape.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data using fitted parameters.

        This method applies the learned transformation to new data.
        It should:
        1. Check that preprocessor is fitted
        2. Validate input shape matches fitted data
        3. Apply transformation
        4. Return transformed array

        Args:
            X: Input data array, shape (n_samples, n_features).

        Returns
        -------
            Transformed data array, shape (n_samples, n_features_out).

        Raises
        ------
            RuntimeError: If preprocessor has not been fitted.
            ValueError: If input shape is incompatible.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse the transformation.

        This method reverses the transformation to recover (approximately)
        the original data space. It should:
        1. Check that preprocessor is fitted
        2. Validate input shape
        3. Apply inverse transformation
        4. Return reconstructed array

        Args:
            X: Transformed data array, shape (n_samples, n_features_out).

        Returns
        -------
            Reconstructed data array, shape (n_samples, n_features).

        Raises
        ------
            RuntimeError: If preprocessor has not been fitted.
            ValueError: If input shape is incompatible.
        """
        raise NotImplementedError
