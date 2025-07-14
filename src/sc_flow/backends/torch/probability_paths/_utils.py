from typing import Any

import torch

from sc_flow._constants import (
    MU_T_FN_KEY,
    SIGMA_T_FN_KEY,
    U_T_FN_KEY,
)
from sc_flow._utils import verify_fn_signature
from sc_flow.backends.torch._types import (
    ProbabilityPathDict,
    ProbabilityPathId,
    TDriftFn,
    TMeanFn,
    TSigmaFn,
)
from sc_flow.backends.torch._utils import broadcast_to_target_shape
from sc_flow.backends.torch.probability_paths._probability_paths import (
    BaseProbabilityPath,
    LinearDiracProbabilityPath,
    LinearGaussianProbabilityPath,
    SchrodingerBridgeProbabilityPath,
    VariancePreservingDiracProbabilityPath,
)

__all__ = [
    "get_probability_path",
    "verify_probability_path_dictionary",
    "make_custom_probability_path",
]


def get_probability_path(
    probability_path_id_or_dict: ProbabilityPathId | ProbabilityPathDict,
    sigma: float,
    require_prng: bool = True,
    prng: torch.Generator | None = None,
    compute_mu_t_kwargs: dict[str, Any] | None = None,
    compute_ut_kwargs: dict[str, Any] | None = None,
    compute_sigma_t_kwargs: dict[str, Any] | None = None,
) -> BaseProbabilityPath:
    r"""Retrieves an instance of :class: `BaseProbabilityPath` from the given settings.

    Probability Paths can be instantiated in two ways:
        * Either by using a string identifier, in which case one will have to specify
            a correct identifier for the pre-defined probability paths. The pre-defined probability paths
            currently supported are:
             * Constant Noise Gaussian Probability Path, instantiated with :param: `probability_path` set to either `"constant-noise-linear-gaussian"` or `"cnlg-pp"`.
             * Schrodinger Bridge Gaussian Probability Path, instantiated with :param: `probability_path` set to either `"schrodinger-bridge-gaussian"` or `"sbg-pp"`.
             * Variance Preserving Dirac Probability Path, instantiated with :param: `probability_path` set to either `"variance-preserving-dirac"` or `"vpd-pp"`.
             * Linear Dirac Probability Path, instantiated with :param: `probability_path` set to either `"linear-dirac"` or `"ld-pp"`.

        * It is also possible to instantiate custom probability paths by passing a dictionary containing the following key-value pairs:
            * `"compute_mu_t"`, mapping to a function taking as input :class: `torch.Tensor`s `t`, `x0` and `x1`
                used to compute the mean of the condiyional probability path at each time step.
            * `"compute_ut"`, mapping to a function taking as input :class: `torch.Tensor`s `t`, `xt`, `x0` and `x1`
                used to compute the velocity field generating the conditional probability path at each time step.
            * `"compute_sigma_t"`, mapping to a function taking as input :class: `torch.Tensor` `t`
                used to compute the noise strength of the probability path at each time step. Optional,
                as it is only required when instantiating non-deterministic custom probability paths (i.e.: :param: `require_prng` is set to `True`).

    :param probability_path_id_or_dict: Either a string identifier among the supported ones (i.e.: check :class: `ProbabilityPathId`) or a
        dictionary with the functions needed for the definition of custom conditional probability paths (i.e.: check :class: `ProbabilityPathDict`).
    :type probability_path_id_or_dict: class: `ProbabilityPathId | ProbabilityPathDict`

    :param sigma: Positive scalar for the noise strength of the probability path.
        This will determine the factor :math: `\sigma` by which the time-dependent standard deviation
        :math: `\sigma_t` of the conditional probability path will be scaled.
        Only used when instantiating non-determinististic probability paths (i.e.: :param: `require_prng` is set to `True`).
    :type sigma: class: `float`

    :param require_prng: Whether a Pseudo-Random Numbers Generator is required for the probability path.
        Pseudo-Random Numbers Generators are required for non-deterministic probability paths and should be instances of
        :class: `torch.Generator`. Only used for custom probability paths retrieved with a
        :class: `ProbabilityPathDict` as :param: `probability_path_id_or_dict`. Defaults to `True`.
    :type require_prng: class: `bool`

    :param prng: Pseudo-Random Numbers Generator used to generate random numbers. Only needed for
        non deterministic probability paths. Only used for custom probability paths retrieved
        with a :class: `ProbabilityPathDict` as :param: `probability_path_id_or_dict`Defaults to `None`.
    :type prng: class: `torch.Generator | None`

    :param compute_mu_t_kwargs: Optional key-word arguments to be passed to the function for computing
        the mean of the conditional probability path. Only used for custom probability paths retrieved
        with a :class: `ProbabilityPathDict` as :param: `probability_path_id_or_dict`. Defaults to `None`.
    :type compute_mu_t_kwargs: class: `dict[str, Any]`

    :param compute_ut_kwargs: Optional key-word arguments to be passed to the function for computing
        the velocity field generating the conditional probability path.
        Only used for custom probability paths retrieved with a :class: `ProbabilityPathDict` as :param: `probability_path_id_or_dict`.
        Defaults to `None`.
    :type compute_ut_kwargs: class: `dict[str, Any]`

    :param compute_sigma_t_kwargs: Optional key-word arguments to be passed to the function for computing
        the noise strength of the conditional probability path.
        Only used for custom probability paths retrieved with a :class: `ProbabilityPathDict` as :param: `probability_path_id_or_dict`
        when :param: `require_prng` is set to `True`.Defaults to `None`.
    :type compute_sigma_t_kwargs: class: `dict[str, Any]`

    :return: Returns the instantiated conditional probability path.
    :type: class: `BaseProbabilityPath`
    """
    if probability_path_id_or_dict in ["constant-noise-linear-gaussian", "cnlg-pp"]:
        return LinearGaussianProbabilityPath(sigma, prng=prng)
    elif probability_path_id_or_dict in ["schrodinger-bridge-gaussian", "sbg-pp"]:
        return SchrodingerBridgeProbabilityPath(sigma, prng=prng)
    elif probability_path_id_or_dict in ["variance-preserving-dirac", "vpd-pp"]:
        return VariancePreservingDiracProbabilityPath(sigma, prng=prng)
    elif probability_path_id_or_dict in ["linear-dirac", "ld-pp"]:
        return LinearDiracProbabilityPath(sigma, prng=prng)
    elif isinstance(probability_path_id_or_dict, dict):
        compute_mu_t_kwargs = {} if compute_mu_t_kwargs is None else compute_mu_t_kwargs
        compute_ut_kwargs = {} if compute_ut_kwargs is None else compute_ut_kwargs
        compute_sigma_t_kwargs = {} if compute_sigma_t_kwargs is None else compute_sigma_t_kwargs

        verify_probability_path_dictionary(
            probability_path_id_or_dict,
            require_prng,
            compute_mu_t_kwargs,
            compute_ut_kwargs,
            compute_sigma_t_kwargs,
        )
        return make_custom_probability_path(
            probability_path_id_or_dict,
            sigma,
            require_prng=require_prng,
            prng=prng,
        )
    else:
        msg = (
            f"Probability path {probability_path_id_or_dict} not supported. It should be either"
            'a string identifier in `["constant-noise-linear-gaussian", "cnlg-pp"'
            '"schrodinger-bridge-gaussian", "sbg-pp", "variance-preserving-dirac", "vpd-pp"'
            '"linear-dirac", "ld-pp"]` or a dictionary in the format determined by'
            "`sc_flow.backends.torch._types.ProbabilityPathDict` data type."
        )
        raise ValueError(msg)


def verify_probability_path_dictionary(
    probability_path_dict: ProbabilityPathDict,
    require_prng: bool,
    compute_mu_t_kwargs: dict[str, Any],
    compute_ut_kwargs: dict[str, Any],
    compute_sigma_t_kwargs: dict[str, Any],
) -> None:
    r"""Verifies a the probability path dictionary and optional key-word arguments.

    Checks that :param: `probability_path_dict` contains the expected keys, :constant: `sc_flow._constants.MU_T_FN_KEY`, :constant: `sc_flow._constants.U_T_FN_KEY`
    and optionally `sc_flow._constants.SIGMA_T_FN_KEY` (when :param: `require_prng` is set to `True`). Otherwise a :class: `KeyError` is raised.
    It also verifies the function signatures against the templates provided in :class: `sc_flow.backends.torch._types.TMeanFn`,
    :class: `sc_flow.backends.torch._types.TDriftFn` and :class: `sc_flow.backends.torch._types.TSigmaFn`

    :param probability_path_id_or_dict: Either a string identifier among the supported ones (i.e.: check :class: `ProbabilityPathId`) or a
        dictionary with the functions needed for the definition of custom conditional probability paths (i.e.: check :class: `ProbabilityPathDict`).
    :type probability_path_id_or_dict: class: `ProbabilityPathId | ProbabilityPathDict`

    :param require_prng: Whether a Pseudo-Random Numbers Generator is required for the probability path.
        Pseudo-Random Numbers Generators are required for non-deterministic probability paths and should be instances of
        :class: `torch.Generator`. Only used for custom probability paths retrieved with a
        :class: `ProbabilityPathDict` as :param: `probability_path_id_or_dict`. Defaults to `True`.
    :type require_prng: class: `bool`

    :param compute_mu_t_kwargs: Optional key-word arguments to be passed to the function for computing
        the mean of the conditional probability path. Only used for custom probability paths retrieved
        with a :class: `ProbabilityPathDict` as :param: `probability_path_id_or_dict`. Defaults to `None`.
    :type compute_mu_t_kwargs: class: `dict[str, Any]`

    :param compute_ut_kwargs: Optional key-word arguments to be passed to the function for computing
        the velocity field generating the conditional probability path.
        Only used for custom probability paths retrieved with a :class: `ProbabilityPathDict` as :param: `probability_path_id_or_dict`.
        Defaults to `None`.
    :type compute_ut_kwargs: class: `dict[str, Any]`

    :param compute_sigma_t_kwargs: Optional key-word arguments to be passed to the function for computing
        the noise strength of the conditional probability path.
        Only used for custom probability paths retrieved with a :class: `ProbabilityPathDict` as :param: `probability_path_id_or_dict`
        when :param: `require_prng` is set to `True`.Defaults to `None`.
    :type compute_sigma_t_kwargs: class: `dict[str, Any]`
    """
    if MU_T_FN_KEY not in probability_path_dict.keys():
        msg = f"Key {MU_T_FN_KEY} not found in probability path dictionary."
        raise KeyError(msg)
    if U_T_FN_KEY not in probability_path_dict.keys():
        msg = f"Key {U_T_FN_KEY} not found in probability path dictionary."
        raise KeyError(msg)
    if SIGMA_T_FN_KEY not in probability_path_dict.keys() and require_prng:
        msg = f"Key {SIGMA_T_FN_KEY} not found in probability path dictionaryand needed when {require_prng=}."
        raise KeyError(msg)

    verify_fn_signature(probability_path_dict[MU_T_FN_KEY], TMeanFn, compute_mu_t_kwargs)
    verify_fn_signature(probability_path_dict[U_T_FN_KEY], TDriftFn, compute_ut_kwargs)
    if require_prng:
        verify_fn_signature(probability_path_dict[SIGMA_T_FN_KEY], TSigmaFn, compute_ut_kwargs)


def make_custom_probability_path(
    probability_path_dict: ProbabilityPathDict,
    sigma: float,
    require_prng: bool = True,
    prng: torch.Generator | None = None,
) -> type[BaseProbabilityPath]:
    r"""Returns an instance of the custom probability path specified by :param: `probability_path_dict`.

    This is done by defining a nested class inheriting from :class: `BaseProbabilityPath`, needed to wrap
    around the two or three functions (depending on whether the conditional probability path is deterministic
    or not) provided in the :param: `probability_path_dict` to make the necessary pre-processing to the
    input tensor in order to ensure their shapes match.

    :param probability_path_id_or_dict: Either a string identifier among the supported ones (i.e.: check :class: `ProbabilityPathId`) or a
        dictionary with the functions needed for the definition of custom conditional probability paths (i.e.: check :class: `ProbabilityPathDict`).
    :type probability_path_id_or_dict: class: `ProbabilityPathId | ProbabilityPathDict`

    :param sigma: Positive scalar for the noise strength of the probability path.
        This will determine the factor :math: `\sigma` by which the time-dependent standard deviation
        :math: `\sigma_t` of the conditional probability path will be scaled.
        Only used when instantiating non-determinististic probability paths (i.e.: :param: `require_prng` is set to `True`).
    :type sigma: class: `float`

    :param require_prng: Whether a Pseudo-Random Numbers Generator is required for the probability path.
        Pseudo-Random Numbers Generators are required for non-deterministic probability paths and should be instances of
        :class: `torch.Generator`. Only used for custom probability paths retrieved with a
        :class: `ProbabilityPathDict` as :param: `probability_path_id_or_dict`. Defaults to `True`.
    :type require_prng: class: `bool`

    :param prng: Pseudo-Random Numbers Generator used to generate random numbers. Only needed for
        non deterministic probability paths. Only used for custom probability paths retrieved
        with a :class: `ProbabilityPathDict` as :param: `probability_path_id_or_dict`Defaults to `None`.
    :type prng: class: `torch.Generator | None`

    :return: Returns the instantiated custom conditional probability path.
    :type: class: `BaseProbabilityPath`
    """
    if require_prng and SIGMA_T_FN_KEY not in probability_path_dict is None:
        msg = (
            "When `require_prng` is set to `True` (i.e.: for non-deterministic"
            "conditional probability paths), the `probability_path_dict` needs to contain"
            "the function needed to compute the noise strength as value under the `sc_flow._constant.SIGMA_T_FN_KEY`"
            "key, not found in the input dictionary."
        )
        raise ValueError(msg)

    class CustomProbabilityPath(BaseProbabilityPath):
        """"""  # noqa

        _require_prng: bool = require_prng

        def __init__(
            self,
            sigma: float,
            prng: torch.Generator | None = None,
        ) -> None:
            """"""  # noqa

            super().__init__(sigma=sigma, prng=prng)
            self._compute_mu_t_fn = probability_path_dict[MU_T_FN_KEY]
            self._compute_ut_fn = probability_path_dict[U_T_FN_KEY]
            if self._require_prng:
                self._compute_sigma_t_fn = probability_path_dict[SIGMA_T_FN_KEY]

        def compute_mu_t(
            self,
            t: torch.Tensor,
            x0: torch.Tensor,
            x1: torch.Tensor,
        ) -> torch.Tensor:
            """"""  # noqa

            t = broadcast_to_target_shape(t, x0.shape)
            self._verify_shapes(x0, x1)
            self._verify_shapes(t, x1)
            return self._compute_mu_t_fn(t, x0, x1)

        def compute_sigma_t(
            self,
            t: torch.Tensor,
        ) -> torch.Tensor:
            """"""  # noqa

            if require_prng:
                return self._compute_sigma_t_fn(t) * self._sigma
            return torch.zeros_like(t)

        def compute_ut(
            self,
            t: torch.Tensor,
            xt: torch.Tensor,
            x0: torch.Tensor,
            x1: torch.Tensor,
        ) -> torch.Tensor:
            """"""  # noqa

            t = broadcast_to_target_shape(t, x0.shape)
            self._verify_shapes(xt, x1)
            self._verify_shapes(x0, x1)
            self._verify_shapes(t, x1)
            return self._compute_ut_fn(t, xt, x0, x1)

    return CustomProbabilityPath(sigma, prng=prng)
