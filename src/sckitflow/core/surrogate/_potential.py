import abc
from functools import cached_property
from math import prod
from typing import Any, Literal

import torch

from sckitflow._random import generators
from sckitflow.core._types import PredictionData, StepData
from sckitflow.core.methods._base import SupportsInference
from sckitflow.data._datamodule import FlowDataModule
from sckitflow.data._dims import DataDimensions
from sckitflow.data._loader import EvalLoader

__all__ = ["SurrogatePotential"]


def _is_mapping(container: Any) -> bool:
    """Checks whether the input object is a mapping.

    A mapping is anything that exposes the `.items()` method.
    """
    # First, we return True if the container is a dictionary.
    if isinstance(container, dict):
        return True
    # Then, we retrieve the items attribute. If None, it means that
    # the container is NOT a mapping. We use `getattr` with `None` as
    # default.
    items_attr = getattr(container, "items", None)
    # We return False when no such attribute is found.
    if not items_attr:
        return False
    # Finally, we check that the method is a callable.
    return callable(items_attr)


def _get_leaf_target_size(step_data: StepData) -> None | int:
    """Infers the target size of step data leaf.

    Infers the size of the target data. First, the condition data is
    used to infer the size, when it is available. Otherwise, falls back
    to the groups metadata (which is by design shared with the source).
    This is required to align the target data with the condition tensor
    -- their presence would cause a shape mismatch if not properly broadcasted.
    Target states are ignored, as they can be absent at inference time.

    :param step_data: The `StepData` which to infer the target leaf size from.
    """
    # We should only pull the ones for the continuous condition
    # covariates that are not optimized over. This is enforced directly in
    # `_align_cond_dict_with_step_data`, to delegate only the shape retrieval
    # to this function.
    target_condition_data = step_data.get("target_condition_data", None)
    # When target condition data is available, we still need to check that it is
    # not an empty dictionary. If this is the case, we retrieve the first element
    # as a reference, and return its leading dimension as target size.
    if target_condition_data is not None and len(target_condition_data):
        ref = next(iter(target_condition_data.values()))
        return ref.shape[0]
    # If the target condition data is not available, proceed by inferring the
    # size from the groups data.
    target_groups_data = step_data.get("target_group_data", None)
    if target_groups_data is not None and len(target_groups_data):
        ref = next(iter(target_groups_data.values()))
        return ref.shape[0]
    # Otherwise, return None because no target data is available.
    return None


def _get_leaf_source_size(step_data: StepData) -> None | int:
    """Infers the source size of step data leaf.

    Infer the size of the source data. We use the source state to infer the
    size, when available. Otherwise, simply return None -- because when
    the source state is not present, the generation surely happens from noise
    and we are not missing anything from the source side.

    :param step_data: The `StepData` which to infer the source leaf size from.
    """
    # Get the source state to retrieve the shape.
    source_state = step_data.get("source_state", None)
    if source_state is not None:
        return source_state.shape[0]
    # When no source state is present, simply return `None`.
    return None


def _get_additional_condition_keys(
    step_data: StepData,
    cond_dict: dict[str, torch.Tensor],
) -> list[str]:
    """Gets a list of condition keys in the step data that are not in the condition dict.

    :param step_data: The `StepData` to retrieve the target condition keys from.
    :param cond_dict: The condition dictionary to retrieve the query condition keys from.
    """
    # First, we collect all the keys from the condition dictionary.
    optimized_keys = sorted(cond_dict)
    # Then, we retrieve the condition data from the `StepData`.
    target_condition_data = step_data.get("target_condition_data", None)
    # We return an empty list, when no condition covariate is present in the data.
    if target_condition_data is None:
        return []
    # Otherwise, we iterate over all the condition keys in the `StepData`, and
    # filter out the ones that are present in the condition dictionary.
    return [key for key in sorted(target_condition_data) if key not in optimized_keys]


def _align_cond_dict_with_step_data(
    step_data: StepData,
    cond_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    r"""Aligns the condition dictionary with the step data.

    This operation is intended to align the condition tensors of shape $(N, D)$ with the
    shapes of the data leaves. When neither size if found, the condition data is returned
    unchanged. On the contrary, denoting by $B_0$ and $B_1$ the source and target size of a leaf,
    the condition data will be expanded and viewed with shape $(B_0\cdot B_1, N, D)$.

    :param step_data: The `StepData` to align the condition dictionary against.
    :param cond_dict: The condition dictionary to align.
    """
    # Get source and target leaf sizes.
    leaf_source_size: int | None = _get_leaf_source_size(step_data)
    leaf_target_size: int | None = _get_leaf_target_size(step_data)
    # Initialize store for all the dimensions to multiply.
    all_dims = [leaf_source_size]
    # We only consider the condition keys from the target condition data
    # that are not present in the condition dictionary.
    additional_keys = _get_additional_condition_keys(step_data, cond_dict)
    # When additional keys are present, we append it to the dims to expand
    if len(additional_keys):
        all_dims.append(leaf_target_size)
    # Compute total number of observations, when the size returns something.
    to_multiply = [i for i in all_dims if i is not None]
    # Fall back to no-operation when there is no leaf data found.
    if not len(to_multiply):
        return cond_dict
    # Get the total number of elements
    tot_size = prod(to_multiply)
    return {k: v.unsqueeze(0).expand(tot_size, *v.shape) for k, v in cond_dict.items()}


def _align_step_data_to_cond_dict(
    step_data: StepData,
    cond_dict: dict[str, torch.Tensor],
) -> StepData:
    """Aligns the step data to the condition dictionary.

    :param step_data: The `StepData` to be aligned.
    :param cond_dict: The condition dictionary to align against. It needs to be
        already aligned with the step data, before calling this function.
    """
    # Get source and target leaf sizes.
    leaf_source_size: int | None = _get_leaf_source_size(step_data)
    leaf_target_size: int | None = _get_leaf_target_size(step_data)
    # We set those to 1 when they are `None`, so that we can handle them cleanly
    S = leaf_source_size if leaf_source_size is not None else 1
    T = leaf_target_size if leaf_target_size is not None else 1
    # We gate the multiplication with the target to occur ONLY when there are additional
    # keys present. If this is not the case, we do not need to expand
    # the target dimension.
    additional_keys = _get_additional_condition_keys(step_data, cond_dict)
    # When additional keys are present, we append it to the dims to expand
    # We multiply the source and target sizes to get the total number of samples.
    if len(additional_keys):
        tot_size = S * T
    else:
        tot_size = S
    # We check that the condition dictionary contains at least on item.
    # The `StopIteration` error is guarded against from the above check,
    # so that we can safely iterate over its values to retrieve our reference
    # tensor.
    if not len(cond_dict):
        raise ValueError("Cannot compute the potential with an empty condition dictionary.")
    ref = next(iter(cond_dict.values()))
    # We also check that the condition array was already effectively expanded.
    # When there is nothing to expand, the conditoin dictionary will still have 2 dimensions --
    # in this case this function collapses to a no-operation and leaves the `StepData` unchanged.
    if ref.ndim == 2:
        return step_data
    # We expect the aligned condition dictionary to have three dimensions. Hence, we raise an
    # error if this is not the case. Something must have gone wrong, hence we write this guard.
    elif ref.ndim != 3:
        raise ValueError(
            f"Condition data has the wrong number of dimensions -- found {ref.ndim} but expected 3."
            "Please, align the condition dictionary with the step data first."
        )
    # Retrieving the number of optimization samples from the reference tensor.
    # After the expansion of the condition dictionary, a new axis is added as leading dimension
    # Hence, we need to retrieve the dimension at index 1 to get the number of samples.
    N = ref.shape[1]

    def _expand_data(x: torch.Tensor, mode: Literal["source", "target"]) -> torch.Tensor:
        """Expands data to be aligned with the condition and the input number of observations.

        The data is of shape (B, rest); to align it, we need to expand it to shape (B*M, N, *rest).


        :param x: The tensor to expand.
        :param mode: The mode to expand the data with. Can be either "source" or "target".
        """
        # We collect the original shape first, before doing any manipulation.
        orig_shape = x.shape
        # We collect the dimension based on the mode.
        # When the mode is "source", we collect the number of target samples.
        if mode == "source":
            M = T
        # When the mode is "target", we collect the number of source samples.
        elif mode == "target":
            M = S
        else:
            raise ValueError(f'Unsupported mode {mode} for alignment. Possible choices are  `"source"` and `"target"`.')
        # We expand the first dimension and add the target size.
        # Then we collapse the first two dimensions so that the dimensionality
        # contract is respected.
        x = x.unsqueeze(0).expand(M, *orig_shape).reshape(tot_size, *orig_shape[1:])
        # Then we need to unsqueeze at dimension 1 and expand dims
        return x.unsqueeze(1).expand(tot_size, N, *orig_shape[1:])

    def _align_container(
        data_container: torch.Tensor | dict[str, torch.Tensor] | None, mode: Literal["source", "target"]
    ) -> torch.Tensor | dict[str, torch.Tensor] | None:
        """Aligns a data container."""
        # If data dictionary is not provided, return an empty dictionary.
        if data_container is None:
            return None
        # When the container is a tensor, directly expand it.
        elif isinstance(data_container, torch.Tensor):
            return _expand_data(data_container, mode)
        # When the container is a mapping, iterate over
        # the items and expand each of them.
        elif _is_mapping(data_container):
            return {k: _expand_data(v, mode) for k, v in data_container.items()}
        # Otherwise, return the data container unchanged (no-op).
        return data_container

    # Now, we can finally update the step data. First, we define a new
    # empty dictionary, that we will repopulate with th aligned data.
    new_step_data: StepData = {}
    for key, value in step_data.items():
        # First, we reassign the value when it is None.abc
        if value is None:
            new_step_data[key] = None
        # The target condition data needs a specific handling, as we
        # do not want to update or expand the condition covariates that are
        # already present in the condition dictionary.
        elif key == "target_condition_data":
            new_cond_data = {}
            for cond_key, cond_data in value.items():
                # If the condition key is present in the condition dictionary,
                # we do not perform any expansion -- it will be anyway overridden
                # by the `_attach_continuous_conditions_to_step_data` call.
                if cond_key in cond_dict:
                    new_cond_data[cond_key] = cond_data
                # All the other target condition covariates need alignment.
                else:
                    new_cond_data[cond_key] = _expand_data(cond_data, "target")
            new_step_data[key] = new_cond_data
        # Align all the source keys in the input StepData.
        elif key.startswith("source_"):
            new_step_data[key] = _align_container(value, "source")
        # Align all the target keys in the input StepData.
        elif key.startswith("target_"):
            new_step_data[key] = _align_container(value, "target")
        # Otherwise, simply store the original value in the updated data.
        else:
            new_step_data[key] = value
    return new_step_data


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
        # This check is repetitive here, because the condition dictionary will have
        # already passed the `_verify_continuous_conditions_dims` test -- hence,
        # it is surely present as a modeled condition. We do it again for
        # semantic reasons, as we want this function to be agnostic of the other
        # check and on how the `StepData` is constructed from the data dimensionalities.
        if cond_key not in condition_covariates:
            raise ValueError(
                f"The condition key {cond_key} does not appear as condition covariate "
                f"in the input step data -- the available keys are {sorted(condition_covariates)}."
            )

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
        ystar = torch.atleast_1d(torch.squeeze(ystar))
        if ystar.ndim != 1:
            raise ValueError(f"Invalid shape for target response, found {ystar.shape}")
        # For the mask, we do the same. Squeeze when available, otherwise
        # initialize it as ones from the target. Then check that the shape matches the
        # one of the target response. Its shape needs to match the one of the target.
        if mask is not None and mask.dtype not in (torch.bool, torch.uint8):
            raise ValueError("mask must be a per-dimension boolean mask, not an index list.")
        mask = torch.atleast_1d(torch.squeeze(mask)) if mask is not None else torch.ones_like(ystar)
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
        for i, (step_data, leaf) in enumerate(self.predict_dl):
            # Iterate over the generators for reproducibility.
            generator, _ = generators(self._seed, i, device=ref.device)
            # Align condition dictionary with step data.
            new_cond_dict = _align_cond_dict_with_step_data(step_data, cond_dict)
            # Align step data with condition dictionary.
            aligned_step_data = _align_step_data_to_cond_dict(step_data, new_cond_dict)
            # Update step data with the input condition dictionary,
            # predict with the inference module.
            new_step_data = _attach_continuous_conditions_to_step_data(aligned_step_data, new_cond_dict)
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
