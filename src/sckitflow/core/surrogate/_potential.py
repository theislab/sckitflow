import abc
from functools import cached_property

import torch
from tqdm import tqdm

from sckitflow._random import generators
from sckitflow.core._types import PredictionData, StepData
from sckitflow.core.methods._base import SupportsInference
from sckitflow.data._datamodule import FlowDataModule
from sckitflow.data._dims import DataDimensions
from sckitflow.data._loader import EvalLoader

__all__ = ["SurrogatePotential"]


def _attach_continuous_conditions_to_step_data(
    step_data: StepData,
    cond_dict: dict[str, torch.Tensor],
) -> StepData:
    """Attaches the condition dictionary to the step data as target conditions covariates.

    Leaves the rest of the fields unchanged, and returns the updated step data. The rest of the
    continuous conditions covariates will be preserved as from the input `StepData`.

    :param step_data: The input `StepData` to update.
    :param cond_dict: The condition dictionary to update the step data with.
    """
    # First we collect the continuous condition covariates from the step data.
    # If the input step data does not contain any continuous covariate,
    # we raise a ValueError.
    condition_covariates = step_data.get("target_condition_data", None)
    # Immediately raise ValueError when there is no condition data.
    if condition_covariates is None:
        raise ValueError("The input `StepData` does not contain any condition covariates.")
    condition_covariates = dict(condition_covariates)

    # We then collect the keys to update, and write the input tensor in place.
    # To do that, we can iterate over the input condition dictionary and update each key.
    for cond_key, new_cond_data in cond_dict.items():
        # We immediately raise an error if the condition key does not appear
        # as condition covariate in the input step data.
        if cond_key not in condition_covariates:
            raise ValueError(
                f"The condition key {cond_key} does not appear as condition covariate "
                f"in the input step data -- the available keys are {sorted(condition_covariates)}."
            )
        # Check that the old and updated data have the same shape.
        old_cond_data = condition_covariates[cond_key]
        old_shape, new_shape = old_cond_data.shape, new_cond_data.shape
        if old_shape != new_shape:
            raise ValueError(f"Shape mismatch at condition key {cond_key} -- got {new_shape} but expected {old_shape}.")

        # Update condition data with new tensors.
        condition_covariates[cond_key] = new_cond_data

    # Update the step data with the new condition covariates and return it.
    new_step_data = {**step_data, "target_condition_data": condition_covariates}
    return new_step_data


def _verify_continuous_conditions_dims(
    dims: DataDimensions,
    cond_dict: dict[str, torch.Tensor],
) -> None:
    """Verifies that the continuous condition dictionary respects the dimensionality contract.

    The condition tensor will be checked against the dimensionality contract over the
    trailing dimensions -- that needs to correspond to the entry with the same key in
    `dims`.

    :param dims: `DataDimensions` containing the contract for the data dimensionalities.
    :param cond_dict: Dictionary of `torch.Tensors` whose dimension to verify.
    """
    # Parse the data dimensionality to get the registry for continuous condition
    # covariates. This can be `None`, so we raise an error immediately when this is the
    # case, as the check is to be done over the continuous covariates.
    ref_continuous_dims: dict[str, int] | None = dims.condition_continuous_dims
    # We immediately raise ValueError when the reference continuous dimensions are None.
    if ref_continuous_dims is None:
        raise ValueError("The reference dimensions do not include continuous conditions covariate.")

    # Iterate over the condition dictionary items and verify the trailing dimensions.
    for cond_key, cond_data in cond_dict.items():
        # Raise error here, because at the moment we only support two-dimensional
        # condition data for optimization.
        if cond_data.ndim != 2:
            raise ValueError(f"Only two-dimensional inputs supported for the moment, but {cond_data.ndim} were found.")

        # Raise error when the condition key does not appear in the dimensionality.
        if cond_key not in ref_continuous_dims:
            raise ValueError(
                f"Condition key {cond_key} does not appear in the reference dimensions {sorted(ref_continuous_dims)}."
            )
        # Raise error when the dimensionalities mismatch
        ref_dim = ref_continuous_dims[cond_key]
        query_dim = cond_data.shape[-1]
        if query_dim != ref_dim:
            raise ValueError(f"Shape mismatch for condition key {cond_key} -- got {query_dim} but expected {ref_dim}.")


class SurrogatePotential(abc.ABC, torch.nn.Module):
    """Abstract class for surrogate potentials.

    Instances must implement the `get_response` and `compute_raw_potential` methods, satisfying
    the contract specified by the method signature. A surrogate potential is defined
    in terms of an inference module, a target response and a context data module.
    As input for the `forward` method, it will take a dictionary of continuous condition covariates.

    Surrogate potentials will be bound to the datamodule that was used to initialize them; they will always
    use prediction data loaders retrieved from the context data module.

    It follows the sequence of three steps:
    1. Query the inference module with the input condition dictionary (`inferer.predict`).
    2. Extract the response tensor from the prediction (`get_response`).
    3. Evaluate the raw potential on the masked response (`compute_raw_potential`).

    In addition, it also allows to mask custom dimensions out from the evaluation of the potential.
    """

    def __init__(
        self,
        inferer: SupportsInference,
        ystar: torch.Tensor,
        datamodule: FlowDataModule,
        mask: torch.Tensor | None = None,
        seed: int = 0,
    ) -> None:
        """Initializes the surrogate potential.

        :param inferer: An instantiated object satisfying the `SupportsInference` protocol.
        :param ystar: A `torch.Tensor` holding the target response for the potential evaluation.
        :param datamodule: The context `FlowDataModule` to run optimization over.
        :param mask: A boolean mask of shape (D,) selecting the response dimensions to evaluate the potential on.
            Coerced to bool; index lists are not supported. Defaults to None, in which case all dimensions are selected.
        :param seed: Integer to seed random number generation. Defaults to `0`.
        """
        # We squeeze the target tensor and expect it to be a flat tensor.
        ystar = torch.squeeze(ystar)
        if ystar.ndim != 1:
            raise ValueError(f"Invalid shape for target response, found {ystar.shape}")
        # For the mask, we do the same. Squeeze when available, otherwise
        # initialize it as ones from the target. Then check that the shape matches the
        # one of the target response. Its shape needs to match the one of the target.
        if mask is not None and mask.dtype not in (torch.bool, torch.uint8):
            raise ValueError("mask must be a per-dimension boolean mask, not an index list.")
        mask = torch.squeeze(mask) if mask is not None else torch.ones_like(ystar)
        mask = mask.bool()
        if mask.shape != ystar.shape:
            raise ValueError(
                "The mask and the target tensor should have the same shape -- "
                f"found mask of shape {mask.shape}, but target has shape {ystar.shape}."
            )

        # Initialize parent class and register tensor buffers.
        super().__init__()
        self.register_buffer("_ystar", ystar)
        self.register_buffer("_mask", mask)

        # Set attributes.
        self._datamodule = datamodule
        self._inferer = inferer
        self._seed = seed

    @abc.abstractmethod
    def get_response(self, pred_data: PredictionData) -> torch.Tensor:
        """Extracts the response tensor from the prediction data.

        :param pred_data: The raw output of ``inferer.predict``.
        :returns: A tensor of shape ``(N, D)``, where ``N`` matches the
            leading dimension of every condition tensor in ``cond_dict`` and
            ``D`` equals ``len(ystar)`` (the unmasked target's last axis).
            Masking with ``self.mask`` along the last axis yields ``(N, D')``,
            which is what ``compute_raw_potential`` receives.
        """

    @abc.abstractmethod
    def compute_raw_potential(self, y: torch.Tensor, ystar: torch.Tensor) -> torch.Tensor:
        """Evaluates the raw potential.

        :param y: Masked predictions, shape ``(N, D')``, where ``N`` is the
            number of optimization samples in the condition dictionary and
            ``D'`` is the number of unmasked response dimensions.
        :param ystar: Masked target response, shape ``(D',)``. Broadcasts
            against ``y``.
        :returns: Per-sample potential, shape ``(N,)``.
        """

    def forward(
        self,
        cond_dict: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Computes the potential value given the input tensor.

        The potential function will be evaluated over all the leaves of the
        context data module. The potential over all leaves is then
        aggregates using the mean.

        :param cond_dict: A dictionary of input tensors, each of shape `(B, D)`.
        """
        # Verify that the condition dictionary satisfies the dimensionality contract.
        _verify_continuous_conditions_dims(self._datamodule.data_dims, cond_dict)
        # Guard that we are not computing the potential over an empty dictionary.
        # Then, infer the number of optimization samples from the shape of the
        # input dictionary.
        if not len(cond_dict):
            raise ValueError("Cannot compute the potential with an empty condition dictionary.")
        # The `StopIteration` error is guarded against from the above check,
        # so that we can safely iterate over its values to retrieve our reference
        # tensor. Furthermore, it is already ensured that the tensors satisfy the
        # data dimensionality contract and have only two dimensions.
        ref = next(iter(cond_dict.values()))
        # Retrieving the number of optimization samples from the reference tensor.
        # It will always be given by its leading dimension. The reason we need the
        # number of optimization samples is to verify the shape of the potential
        # over each leave -- we should have a potential value for each optimization sample.
        N = ref.shape[0]
        # Verify that all the other keys have the same number of samples
        # and that they are on the same device.
        for cond_key, cond_data in cond_dict.items():
            if cond_data.shape[0] != N:
                raise ValueError(
                    f"Condition data at key {cond_key} has the wrong "
                    f"number of optimization samples -- found {cond_data.shape[0]} "
                    f"but expected {N}."
                )
            if cond_data.device != ref.device:
                raise ValueError(
                    f"Condition data at key {cond_key} is on the wrong device -- "
                    f"found {cond_data.device} but expected {ref.device}."
                )

        # List to store all the computed potentials.
        potentials = []
        # Iterate over the prediction data loader.
        for i, (step_data, leaf) in enumerate(tqdm(self.predict_dl, total=len(self.predict_dl), desc="Predicting")):
            # Iterate over the generators for reproducibility.
            generator, _ = generators(self._seed, i, device=ref.device)
            # Update step data with the input condition dictionary,
            # predict with the inference module.
            new_step_data = _attach_continuous_conditions_to_step_data(step_data, cond_dict)
            pred_data: PredictionData = self.inferer.predict(new_step_data, generator=generator)
            y: torch.Tensor = self.get_response(pred_data)
            # Check that the response has the same dimension as the target
            target_dim = self.ystar.shape[-1]
            if y.shape[-1] != target_dim:
                raise ValueError(
                    f"get_response returned a tensor whose last dim is "
                    f"{y.shape[-1]}, but the target response has dim {target_dim}."
                )

            # Evaluate raw potential. When calling the
            # `compute_raw_potential` method, mask the dimensions that we
            # do not want to consider for the gradient computations.
            pot = self.compute_raw_potential(y[..., self._mask], self.ystar[self._mask])

            # Update the potential values list. Before doing that, verify that
            # the potential value has the correct shape, before proceeding.
            # Otherwise, raise a `ValueError`.
            if pot.shape != (N,):
                raise ValueError(
                    f"Potential value for leaf {leaf} has the wrong shape -- got {pot.shape}, but expected {(N,)}."
                )
            potentials.append(pot)

        # Stack the tensors from the input list and aggregate with the mean.
        # This is similar to a MC average over the considered prediction leaves.
        # We stack and aggregate over the leading dimensions, so that
        # the shape of the sample-wise potential is preserved.
        return torch.mean(torch.stack(potentials, dim=0), dim=0)

    @cached_property
    def predict_dl(self) -> EvalLoader:
        """Caches the `predict_dataloader` call on the context datamodule."""
        return self._datamodule.predict_dataloader()

    @property
    def datamodule(self) -> FlowDataModule:
        """Returns the underlying data module for the context."""
        return self._datamodule

    @property
    def inferer(self) -> SupportsInference:
        """Returns the underlying forward model used to compute the potential."""
        return self._inferer

    @property
    def mask(self) -> torch.Tensor:
        """Returns the mask for the considered response axes."""
        return self._mask

    @property
    def ystar(self) -> torch.Tensor:
        """Returns the target response."""
        return self._ystar

    @property
    def seed(self) -> int:
        """Returns the seed to control random number generation."""
        return self._seed
