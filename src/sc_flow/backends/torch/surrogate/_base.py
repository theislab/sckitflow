import abc
from collections.abc import Collection
from typing import Literal

import torch

__all__ = ["SurrogatePotential"]


class SurrogatePotential(abc.ABC, torch.nn.Module):
    """Base class for surrogate potentials.

    Derived classes should override the `self._compute_raw_potential` method.
    They can specify custom aggregation dimensions through the following class attributes.

    :param _REDUCE_DIMS_INPUT_REGISTRY: Registry mapping each input reduction type to the respective dimensions.
    :type _REDUCE_DIMS_INPUT_REGISTRY: class: `dict[str, int | Collection[int] | None]`

    :param _REDUCE_DIMS_OUTPUT_REGISTRY: Registry mapping each output reduction type to the respective dimensions.
    :type _REDUCE_DIMS_OUTPUT_REGISTRY: class: `dict[str, int | Collection[int] | None]`

    :param _VALID_INPUT_REDUCTION: List of valid reduction identifiers for input responses.
    :type _VALID_INPUT_REDUCTION: class: `Collection[str]`

    :param _VALID_OUTPUT_REDUCTION: List of valid reduction identifiers for output potentials.
    :type _VALID_OUTPUT_REDUCTION: class: `Collection[str]`
    """

    _REDUCE_DIMS_INPUT_REGISTRY = {
        "mean": 0,
        "none": None,
    }
    _REDUCE_DIMS_OUTPUT_REGISTRY = {
        "mean": 0,
        "none": None,
    }
    _VALID_INPUT_REDUCTION = ["mean", "none"]
    _VALID_OUTPUT_REDUCTION = ["mean", "none"]

    def __init__(
        self,
        ystar: torch.Tensor,
        surr_model: torch.nn.Module,
        *args,
        reduction_input: Literal["mean", "none"] | None = None,
        reduction_output: Literal["mean", "none"] | None = None,
        mask: Collection[bool] | None = None,
        **kwargs,
    ) -> None:
        """Initializes the potential given the specified aggregation dimensions.

        :param ystar: The target response to steer for the inverse optimization.
            Can be have any arbitrary number of singleton dimensions, and only an
            additional one with arbitrary size.
        :type ystar: class: `torch.Tensor`

        :param surr_model: The surrogate model to compute the response from.
        :type surr_model: class: `torch.nn.Module`

        :param reduction_input: The aggregation type for the input predictions.
            Defaults to `None`, in which case no aggregation will be performed on the input.
        :type reduction_input: class: `Literal["mean", "none"] | None`

        :param reduction_output: The aggregation type for the output potential.
            Defaults to `None`, in which mean aggregation will be performed on the output potential.
        :type reduction_output: class: `Literal["mean", "none"] | None`
        """
        # ---- 0. Initialize base class ----
        super().__init__()

        # ---- 1. Set attributes ----
        self._reduction_input = "none" if reduction_input is None else reduction_input
        self._reduction_output = "mean" if reduction_output is None else reduction_output
        self._surr_model = surr_model

        # ---- 2. Register buffers ---
        self.register_buffer("_ystar", torch.squeeze(ystar))

        if mask is None:
            mask = torch.ones((self._response_dim,), dtype=torch.bool)
        else:
            mask = torch.as_tensor(mask, dtype=torch.bool)
        self.register_buffer("_mask", torch.squeeze(mask))

        # ---- 3. Verify attributes ----
        if self._ystar.ndim != 1:
            raise ValueError(f"Invalid shape for target response, found {self._ystar.shape}")
        if self._mask.ndim != 1 or self._mask.shape[0] != self._response_dim:
            raise ValueError(
                f"Invalid shape for features mask, found {self._mask.shape[0]} but {self._response_dim} expected."
            )
        if self._reduction_input not in self._VALID_INPUT_REDUCTION:
            raise ValueError(
                f"Input reduction type {self._reduction_input} not supported, possible choices {self._VALID_INPUT_REDUCTION}"
            )
        if self._reduction_output not in self._VALID_OUTPUT_REDUCTION:
            raise ValueError(
                f"Output reduction type {self._reduction_output} not supported, possible choices {self._VALID_OUTPUT_REDUCTION}"
            )

    @property
    def _input_reduction_dims(self) -> int | Collection[int] | None:
        return self._REDUCE_DIMS_INPUT_REGISTRY[self._reduction_input]

    @property
    def _output_reduction_dims(self) -> int | Collection[int] | None:
        return self._REDUCE_DIMS_OUTPUT_REGISTRY[self._reduction_output]

    @property
    def _response_dim(self) -> int:
        return self._ystar.shape[0]

    @abc.abstractmethod
    def _compute_raw_potential(self, y: torch.Tensor) -> torch.Tensor:
        """Computes the raw potential from the input response.

        Needs to be overridden by derived classes.

        :param y: The input response from the surrogate model to evaluate the potential on.
        :type y: class: `torch.Tensor`
        """

    def _verify_input_shape(self, x: torch.Tensor) -> None:
        """Verifies the shape of the input to the surrogate model before evaluation.

        :param x: The input to verify on which to evaluate the surrogate model.
        :type x: class: `torch.Tensor`
        """
        # ---- 1. Verify that we only have two dimensions ----
        if x.ndim != 2:
            raise ValueError(f"Only two-dimensional inputs supported for the moment, but {x.ndim} were found.")

    def _verify_response_shape(self, y: torch.Tensor, N: int) -> None:
        """Verifies the shape of the input surrogate response before reduction.

        :param y: The input response from the surrogate model whose shape to verify.
        :type y: class: `torch.Tensor`

        :param N: The expected number of samples determined by the input to calculate
            the response. When reduction is "none", the response should be of shape (N, G).
            When reduction is "mean", it should be of shape (M, N, G), with M being the
            number of samples to reduce over.
        :type N: class: `int`
        """
        # ---- 1. Verify that the dimensionality matches with the target response ----
        if y.shape[-1] != self._response_dim:
            raise ValueError(
                f"The evaluated response does not match with the supplied target, "
                f"found response of shape {y.shape}, but expected dimensionality to be {self._response_dim}"
            )

        # ---- 2. Verify that the aggregation dimensionalities are correct when no reduction used ----
        if self._reduction_input == "none":
            # ---- 2.1 Check that the number of dimensions is correct ----
            if y.ndim != 2:
                raise ValueError(
                    f'When input reduction is "none", the response should have two dimensions, but {y.ndim} were found.'
                )

            # ---- 2.2 Check that the batch dimensionality is correct ----
            B, _ = y.shape
            if B != N:
                raise ValueError(
                    f"Response should have the same batch dimension as the input, found {B} but expected {N}."
                )

        # ---- 3. Verify that the aggregation dimensionalities are correct when mean reduction used ----
        elif self._reduction_input == "mean":
            # ---- 3.1 Check that the number of dimensions is correct ----
            if y.ndim != 3:
                raise ValueError(
                    f'When input reduction is "mean", the response should have three dimensions, but {y.ndim} were found.'
                )

            # ---- 3.2 Check that the batch dimensionality is correct ----
            _, B, _ = y.shape
            if B != N:
                raise ValueError(
                    f"Response should have the same batch dimension as the input, found {B} but expected {N}."
                )
        else:
            raise ValueError(
                f"Input reduction type {self._reduction_input} not supported, possible choices {self._VALID_INPUT_REDUCTION}"
            )

    def _verify_potential_shape(self, psi: torch.Tensor, N: int) -> None:
        """Verifies the shape of the output surrogate potential before aggregation.

        :param psi: The output surrogate potential whose shape to verify.
        :type psi: class: `torch.Tensor`

        :param N: The expected number of samples determined by the input to calculate
            the response. When reduction is "none", the response should be of shape (N,).
            When reduction is "mean", it should be of shape (M, N), with M being the
            number of samples to reduce over.
        :type N: class: `int`
        """
        # ---- 1. Verify that the number of samples match when no reduction is used ----
        if self._reduction_output == "none":
            # ---- 1.1 Only one dimension expected ----
            if psi.ndim != 1:
                raise ValueError(
                    f'When output reduction is "none", the potential should have one dimension, but {psi.ndim} were found.'
                )

            # ----- 1.2 The dimension should match the number of samples ----
            B = psi.shape[0]
            if B != N:
                raise ValueError(
                    f"Potential should have the same batch dimension as the input, found {B} but expected {N}."
                )

        # ---- 2. Verify that the number of samples match when mean reduction is used ----
        if self._reduction_output == "mean":
            # ---- 2.1 Two dimensions expected ----
            if psi.ndim != 2:
                raise ValueError(
                    f'When output reduction is "mean", the potential should have two dimensions, but {psi.ndim} were found.'
                )

            # ----- 2.2 The batch dimension should match the number of samples ----
            _, B = psi.shape
            if B != N:
                raise ValueError(
                    f"Potential should have the same batch dimension as the input, found {B} but expected {N}."
                )

    def _query_surr_model(self, x: torch.Tensor) -> torch.Tensor:
        """Queries the underlying surrogate model from the input data.

        :param x: The input data to query the forward model on.
        :type x: class: `torch.Tensor`
        """
        return self._surr_model(x)

    def _get_reduced_response(self, y: torch.Tensor) -> torch.Tensor:
        """Reduces the input tensor before computing the potential.

        :param y: The input response from the surrogate model to reduce.
        :type y: class: `torch.Tensor`
        """
        # ---- 1. Early return with no reduction ----
        if self._reduction_input == "none":
            return y

        # ---- 2. Get reduction dimensionality and reduce with mean ----
        elif self._reduction_input == "mean":
            reduce_dim = self._input_reduction_dims
            return torch.mean(y, dim=reduce_dim)

        # ---- 3. Otherwise throw a ValueError if the cases are not matched ----
        else:
            raise ValueError(
                f"Input reduction type {self._reduction_input} not supported, possible choices {self._VALID_INPUT_REDUCTION}"
            )

    def _get_reduced_potential(self, psi: torch.Tensor) -> torch.Tensor:
        """Reduces the output surrogate potential value before returning it.

        :param psi: The output surrogate potential value to reduce.
        :type psi: class: `torch.Tensor`
        """
        # ---- 1. Early return with no reduction ----
        if self._reduction_output == "none":
            return psi

        # ---- 2. Get reduction dimensionality and reduce with mean ----
        elif self._reduction_output == "mean":
            reduce_dim = self._output_reduction_dims
            return torch.mean(psi, dim=reduce_dim)

        # ---- 3. Otherwise throw a ValueError if the cases are not matched ----
        else:
            raise ValueError(
                f"Output reduction type {self._reduction_output} not supported, possible choices {self._VALID_OUTPUT_REDUCTION}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the potential value given the input tensor.

        :param x: The input tensor of shape `(B, D)`.
        :type x: class: `torch.Tensor`
        """
        # ---- 0. Get dimensionalities of input x ----
        self._verify_input_shape(x)
        N, D = x.shape

        # ---- 1. Query surrogate model and reduce response ----
        y = self._query_surr_model(x)
        self._verify_response_shape(y, N)
        y_red = self._get_reduced_response(y)

        # ---- 2. Compute raw potential and reduce it ----
        psi = self._compute_raw_potential(y_red)
        self._verify_potential_shape(psi, N)
        return self._get_reduced_potential(psi)

    @property
    def mask(self) -> torch.Tensor:
        """Returns the mask for the considered response axes."""
        return self._mask

    @property
    def ystar(self) -> torch.Tensor:
        """Returns the target response."""
        return self._ystar

    @property
    def surr_model(self) -> torch.nn.Module:
        """Returns the underlying surrogate model."""
        return self._surr_model
