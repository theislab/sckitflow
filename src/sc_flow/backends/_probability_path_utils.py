from collections.abc import Callable
from functools import partial
from typing import Any

from sc_flow._constants import (
    MU_T_FN_KEY,
    SIGMA_T_FN_KEY,
    U_T_FN_KEY,
)
from sc_flow._runtime import (
    raise_runtime_error_on_backend_failed_import,
    raise_runtime_error_on_backend_not_supported,
    set_jax_import_failed,
    set_torch_import_failed,
)
from sc_flow._types import ProbabilityPathId
from sc_flow._utils import verify_fn_signature

try:
    from torch import Generator

    from sc_flow.backends.torch._types import (
        ProbabilityPathDict as TorchProbabilityDict,
    )
    from sc_flow.backends.torch.probability_paths._probability_paths import (
        BaseProbabilityPath as TorchBaseProbabilityPath,
    )
except (ImportError, ModuleNotFoundError):
    set_torch_import_failed(True)
    Generator = None
    TorchProbabilityDict = None
    TorchBaseProbabilityPath = None

try:
    from sc_flow.backends.jax._types import (
        ProbabilityPathDict as JaxProbabilityDict,
    )
    from sc_flow.backends.jax.probability_paths._probability_paths import BaseProbabilityPath as JaxBaseProbabilityPath
except (ImportError, ModuleNotFoundError):
    set_jax_import_failed(True)
    JaxProbabilityDict = None
    JaxBaseProbabilityPath = None


__all__ = [
    "get_probability_path",
    "make_custom_probability_path",
]


def get_probability_path(
    sigma: float,
    probability_path_id_or_dict: ProbabilityPathId | JaxProbabilityDict | TorchProbabilityDict | None = None,
    compute_mu_t_fn: Callable | None = None,
    compute_ut_fn: Callable | None = None,
    compute_sigma_t_fn: Callable | None = None,
    require_prng: bool = True,
    prng: Generator | None = None,
    compute_mu_t_kwargs: dict[str, Any] | None = None,
    compute_ut_kwargs: dict[str, Any] | None = None,
    compute_sigma_t_kwargs: dict[str, Any] | None = None,
) -> TorchBaseProbabilityPath | JaxBaseProbabilityPath:
    r"""Retrieves an instance of :class: `BaseProbabilityPath` from the given settings.

    Probability Paths can be instantiated in three ways:
        * Either by using a string identifier, in which case one will have to specify
            a correct identifier for the pre-defined probability paths. The pre-defined probability paths
            currently supported are:
             * Constant Noise Gaussian Probability Path, instantiated with :param: `probability_path_id_or_dict` set to either `"constant-noise-linear-gaussian"` or `"cnlg-pp"`.
             * Schrodinger Bridge Gaussian Probability Path, instantiated with :param: `probability_path_id_or_dict` set to either `"schrodinger-bridge-gaussian"` or `"sbg-pp"`.
             * Variance Preserving Dirac Probability Path, instantiated with :param: `probability_path_id_or_dict` set to either `"variance-preserving-dirac"` or `"vpd-pp"`.
             * Linear Dirac Probability Path, instantiated with :param: `probability_path_id_or_dict` set to either `"linear-dirac"` or `"ld-pp"`.

        * It is also possible to instantiate custom probability paths by passing a dictionary containing the following key-value pairs:
            * `"compute_mu_t"`, mapping to a function taking as input `t`, `x0` and `x1`
                used to compute the mean of the conditional probability path at each time step.
            * `"compute_ut"`, mapping to a function taking as input `t`, `xt`, `x0` and `x1`
                used to compute the velocity field generating the conditional probability path at each time step.
            * `"compute_sigma_t"`, mapping to a function taking as input `t`
                used to compute the noise strength of the probability path at each time step. Optional,
                as it is only required when instantiating non-deterministic custom probability paths (i.e.: :param: `require_prng` is set to `True`).
            Such functions need to accept their input as either :class: `torch.Tensor` or :class: `jax.Array` depending on the chosen backend.

        * Alternatively, one can also pass the needed functions as the :param: `compute_mu_t_fn`, :param: `compute_ut_fn` and :param: `compute_sigma_t_fn`,
            which then will be used to compute the conditional probability path. These will have to satisfy the aforementioned requisites too.

    :param sigma: Positive scalar for the noise strength of the probability path.
        This will determine the factor :math: `\sigma` by which the time-dependent standard deviation
        :math: `\sigma_t` of the conditional probability path will be scaled.
        Only used when instantiating non-determinististic probability paths (i.e.: :param: `require_prng` is set to `True`).
    :type sigma: class: `float`

    :param probability_path_id_or_dict: Either a string identifier among the supported ones (i.e.: check :class: `ProbabilityPathId`) or a
        dictionary with the functions needed for the definition of custom conditional probability paths (i.e.: check :class: `ProbabilityPathDict`).
        When `None`, you need to pass the functions as arguments in :param: `compute_mu_t_fn`, :param: `compute_ut_fn` and :param: `compute_sigma_t_fn`.
        A :class: `ValueError` is thrown otherwise. Setting the probability path with :param: `probability_path_id_or_dict` has priority over
        doing it with the individual functions in :param: `compute_mu_t_fn`, :param: `compute_ut_fn` and :param: `compute_sigma_t_fn`,
        which are ignored otherwise. Defaults to `None`.
    :type probability_path_id_or_dict: class: `ProbabilityPathId | ProbabilityPathDict | None`

    :param compute_mu_t_fn: The function for computing the mean of the conditional probability path.
        Only used for custom probability paths when :param: `probability_path_id_or_dict` is `None`
        and is ignored otherwise. When passed, it has to match the template defined in :class: `sc_flow.backends.torch._types.TMeanFn`. Defaults to `None`.
    :type compute_mu_t_fn: class: `TMeanFn | None`

    :param compute_ut_fn: The function for computing the velocity field generating the conditional probability path.
        Only used for custom probability paths when :param: `probability_path_id_or_dict` is `None`
        and is ignored otherwise. When passed, it has to match the template defined in :class: `sc_flow.backends.torch._types.TDriftFn`. Defaults to `None`.
    :type compute_mu_t_fn: class: `TDriftFn | None`

    :param compute_sigma_t_fn: The function for computing the noise scale for the conditional probability path.
        Only used for custom probability paths when :param: `probability_path_id_or_dict` is `None` and :param: `require_prng` is `True`
        and is ignored otherwise. When passed, it has to match the template defined in :class: `sc_flow.backends.torch._types.TSigmaFn`. Defaults to `None`.
    :type compute_mu_t_fn: class: `TSigmaFn | None`

    :param require_prng: Whether a Pseudo-Random Numbers Generator is required for the probability path.
        Pseudo-Random Numbers Generators are required for non-deterministic probability paths and should be instances of
        :class: `torch.Generator`. Only used for custom probability paths retrieved with a
        :class: `ProbabilityPathDict` as :param: `probability_path_id_or_dict`. Defaults to `True`.
    :type require_prng: class: `bool`

    :param prng: Pseudo-Random Numbers Generator used to generate random numbers. Only needed for
        non deterministic probability paths. Only used for custom probability paths retrieved
        with a :class: `ProbabilityPathDict` as :param: `probability_path_id_or_dict` or by setting
        the :param: `compute_mu_t_fn`, :param: `compute_ut_fn` and :param: `compute_sigma_t_fn` arguments. Defaults to `None`.
    :type prng: class: `torch.Generator | None`

    :param compute_mu_t_kwargs: Optional key-word arguments to be passed to the function for computing
        the mean of the conditional probability path. Only used for custom probability paths retrieved
        with a :class: `ProbabilityPathDict` as :param: `probability_path_id_or_dict` or by setting
        the :param: `compute_mu_t_fn`, :param: `compute_ut_fn` and :param: `compute_sigma_t_fn` arguments. Defaults to `None`.
    :type compute_mu_t_kwargs: class: `dict[str, Any] | None`

    :param compute_ut_kwargs: Optional key-word arguments to be passed to the function for computing
        the velocity field generating the conditional probability path.
        Only used for custom probability paths retrieved with a :class: `ProbabilityPathDict` as :param: `probability_path_id_or_dict` or by setting
        the :param: `compute_mu_t_fn`, :param: `compute_ut_fn` and :param: `compute_sigma_t_fn` arguments.
        Defaults to `None`.
    :type compute_ut_kwargs: class: `dict[str, Any] | None`

    :param compute_sigma_t_kwargs: Optional key-word arguments to be passed to the function for computing
        the noise strength of the conditional probability path.
        Only used for custom probability paths retrieved with a :class: `ProbabilityPathDict` as :param: `probability_path_id_or_dict` or by setting
        the :param: `compute_mu_t_fn`, :param: `compute_ut_fn` and :param: `compute_sigma_t_fn` arguments,
        when :param: `require_prng` is set to `True`. Defaults to `None`.
    :type compute_sigma_t_kwargs: class: `dict[str, Any] | None`

    :return: Returns the instantiated conditional probability path.
    :type: class: `BaseProbabilityPath`
    """
    from sc_flow._runtime import BACKEND

    raise_runtime_error_on_backend_failed_import()

    if probability_path_id_or_dict in ["constant-noise-linear-gaussian", "cnlg-pp"]:
        if BACKEND == "torch":
            from sc_flow.backends.torch.probability_paths._probability_paths import LinearGaussianProbabilityPath

            return LinearGaussianProbabilityPath(sigma, prng=prng)
        elif BACKEND == "jax":
            from sc_flow.backends.jax.probability_paths._probability_paths import LinearGaussianProbabilityPath

            return LinearGaussianProbabilityPath(sigma)
        else:
            raise_runtime_error_on_backend_not_supported(BACKEND)

    elif probability_path_id_or_dict in ["schrodinger-bridge-gaussian", "sbg-pp"]:
        if BACKEND == "torch":
            from sc_flow.backends.torch.probability_paths._probability_paths import SchrodingerBridgeProbabilityPath

            return SchrodingerBridgeProbabilityPath(sigma, prng=prng)
        elif BACKEND == "jax":
            from sc_flow.backends.jax.probability_paths._probability_paths import SchrodingerBridgeProbabilityPath

            return SchrodingerBridgeProbabilityPath(sigma)
        else:
            raise_runtime_error_on_backend_not_supported(BACKEND)

    elif probability_path_id_or_dict in ["variance-preserving-dirac", "vpd-pp"]:
        if BACKEND == "torch":
            from sc_flow.backends.torch.probability_paths._probability_paths import (
                VariancePreservingDiracProbabilityPath,
            )

            return VariancePreservingDiracProbabilityPath(sigma, prng=prng)
        elif BACKEND == "jax":
            from sc_flow.backends.jax.probability_paths._probability_paths import VariancePreservingDiracProbabilityPath

            return VariancePreservingDiracProbabilityPath(sigma)
        else:
            raise_runtime_error_on_backend_not_supported(BACKEND)

    elif probability_path_id_or_dict in ["linear-dirac", "ld-pp"]:
        if BACKEND == "torch":
            from sc_flow.backends.torch.probability_paths._probability_paths import LinearDiracProbabilityPath

            return LinearDiracProbabilityPath(sigma, prng=prng)
        elif BACKEND == "jax":
            from sc_flow.backends.jax.probability_paths._probability_paths import LinearDiracProbabilityPath

            return LinearDiracProbabilityPath(sigma)
        else:
            raise_runtime_error_on_backend_not_supported(BACKEND)

    elif isinstance(probability_path_id_or_dict, dict):
        if MU_T_FN_KEY not in probability_path_id_or_dict.keys():
            msg = f"Key {MU_T_FN_KEY} not found in probability path dictionary."
            raise KeyError(msg)
        if U_T_FN_KEY not in probability_path_id_or_dict.keys():
            msg = f"Key {U_T_FN_KEY} not found in probability path dictionary."
            raise KeyError(msg)
        if SIGMA_T_FN_KEY not in probability_path_id_or_dict.keys() and require_prng:
            msg = f"Key {SIGMA_T_FN_KEY} not found in probability path dictionary and needed when {require_prng=}."
            raise KeyError(msg)
        return make_custom_probability_path(
            sigma,
            probability_path_id_or_dict.get(MU_T_FN_KEY),
            probability_path_id_or_dict.get(U_T_FN_KEY),
            probability_path_id_or_dict.get(SIGMA_T_FN_KEY, None),
            require_prng=require_prng,
            prng=prng,
            compute_mu_t_kwargs=compute_mu_t_kwargs,
            compute_ut_kwargs=compute_ut_kwargs,
            compute_sigma_t_kwargs=compute_sigma_t_kwargs,
        )

    elif probability_path_id_or_dict is None:
        return make_custom_probability_path(
            sigma,
            compute_mu_t_fn,
            compute_ut_fn,
            compute_sigma_t_fn,
            require_prng=require_prng,
            prng=prng,
            compute_mu_t_kwargs=compute_mu_t_kwargs,
            compute_ut_kwargs=compute_ut_kwargs,
            compute_sigma_t_kwargs=compute_sigma_t_kwargs,
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


def make_custom_probability_path(
    sigma: float,
    compute_mu_t_fn: Callable,
    compute_ut_fn: Callable,
    compute_sigma_t_fn: Callable | None,
    require_prng: bool = True,
    prng: Generator | None = None,
    compute_mu_t_kwargs: dict[str, Any] | None = None,
    compute_ut_kwargs: dict[str, Any] | None = None,
    compute_sigma_t_kwargs: dict[str, Any] | None = None,
) -> TorchBaseProbabilityPath | JaxBaseProbabilityPath:
    r"""Returns an instance of the custom probability path specified by functions passed as argument.

    This is done by defining a nested class inheriting from :class: `BaseProbabilityPath`, needed to wrap
    around the two or three functions (depending on whether the conditional probability path is deterministic
    or not) provided in the :param: `probability_path_dict` to make the necessary pre-processing to the
    input tensor in order to ensure their shapes match.

    :param sigma: Positive scalar for the noise strength of the probability path.
        This will determine the factor :math: `\sigma` by which the time-dependent standard deviation
        :math: `\sigma_t` of the conditional probability path will be scaled.
        Only used when instantiating non-determinististic probability paths (i.e.: :param: `require_prng` is set to `True`).
    :type sigma: class: `float`

    :param compute_mu_t_fn: The function for computing the mean of the conditional probability path.
        It has to match the template defined in :class: `sc_flow.backends.torch._typesTMeanFn`.
    :type compute_mu_t_fn: class: `TMeanFn`

    :param compute_ut_fn: The function for computing the velocity field generating the conditional probability path.
        It has to match the template defined in :class: `sc_flow.backends.torch._typesTDriftFn`.
    :type compute_mu_t_fn: class: `TDriftFn`

    :param compute_sigma_t_fn: The function for computing the velocity field generating the conditional probability path.
        Only used when :param: `require_prng` is `True` and is ignored otherwise.
        When passed it has to match the template defined in :class: `sc_flow.backends.torch._typesTSigmaFn`. Defaults to `None`.
    :type compute_mu_t_fn: class: `TSigmaFn | None`

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
        the mean of the conditional probability path.
    :type compute_mu_t_kwargs: class: `dict[str, Any]`

    :param compute_ut_kwargs: Optional key-word arguments to be passed to the function for computing
        the velocity field generating the conditional probability path.
    :type compute_ut_kwargs: class: `dict[str, Any]`

    :param compute_sigma_t_kwargs: Optional key-word arguments to be passed to the function for computing
        the noise strength of the conditional probability path. Only used when :param: `require_prng` is set to `True`.
    :type compute_sigma_t_kwargs: class: `dict[str, Any]`

    :return: Returns the instantiated custom conditional probability path.
    :type: class: `BaseProbabilityPath`
    """
    from sc_flow._runtime import BACKEND

    if BACKEND == "torch":
        try:
            from torch import zeros_like

            from sc_flow.backends.torch._types import TDriftFn, TMeanFn, TSigmaFn
            from sc_flow.backends.torch._utils import broadcast_to_target_shape

            BaseProbabilityPath = TorchBaseProbabilityPath
        except (ImportError, ModuleNotFoundError):
            set_torch_import_failed(True)
            raise_runtime_error_on_backend_failed_import()
    elif BACKEND == "jax":
        try:
            from jax.numpy import zeros_like

            from sc_flow.backends.jax._types import TDriftFn, TMeanFn, TSigmaFn
            from sc_flow.backends.jax._utils import broadcast_to_target_shape

            BaseProbabilityPath = JaxBaseProbabilityPath
        except (ImportError, ModuleNotFoundError):
            set_jax_import_failed(True)
            raise_runtime_error_on_backend_failed_import()
    else:
        raise_runtime_error_on_backend_not_supported(BACKEND)

    if compute_mu_t_fn is None:
        msg = "When probability_path_id_or_dict is `None`, you need to pass a correct `compute_mu_t_fn` as argument, found `None`."
        raise ValueError(msg)
    if compute_ut_fn is None:
        msg = "When probability_path_id_or_dict is `None`, you need to pass a correct `compute_ut_fn` as argument, found `None`."
        raise ValueError(msg)
    if require_prng and compute_sigma_t_fn is None:
        msg = (
            "When probability_path_id_or_dict is `None`, you need to pass a correct `compute_sigma_t_fn` as argument "
            "when initializing custom non-deterministic conditional probability paths, found `None`."
        )
        raise ValueError(msg)

    compute_mu_t_kwargs = {} if compute_mu_t_kwargs is None else compute_mu_t_kwargs
    compute_ut_kwargs = {} if compute_ut_kwargs is None else compute_ut_kwargs
    compute_sigma_t_kwargs = {} if compute_sigma_t_kwargs is None else compute_sigma_t_kwargs

    verify_fn_signature(compute_mu_t_fn, TMeanFn, compute_mu_t_kwargs)
    compute_mu_t_fn = partial(compute_mu_t_fn, **compute_mu_t_kwargs)

    verify_fn_signature(compute_ut_fn, TDriftFn, compute_ut_kwargs)
    compute_ut_fn = partial(compute_ut_fn, **compute_ut_kwargs)

    if require_prng:
        verify_fn_signature(compute_sigma_t_fn, TSigmaFn, compute_sigma_t_kwargs)
        compute_sigma_t_fn = partial(compute_sigma_t_fn, **compute_sigma_t_kwargs)
    else:
        compute_sigma_t_fn = None

    if require_prng and compute_sigma_t_fn is None:
        msg = (
            "When probability_path_id_or_dict is `None`, you need to pass a correct `compute_sigma_t_fn` as argument"
            "when initializing custom non-deterministic conditional probability paths, found `None`."
        )
        raise ValueError(msg)

    class CustomProbabilityPath(BaseProbabilityPath):
        """"""  # noqa

        _require_prng: bool = require_prng

        def __init__(
            self,
            sigma,
            prng,
        ) -> None:
            """"""  # noqa
            if BACKEND == "torch":
                super().__init__(sigma=sigma, prng=prng)
            else:
                super().__init__(sigma=sigma)
            self._compute_mu_t_fn = compute_mu_t_fn
            self._compute_ut_fn = compute_ut_fn
            self._compute_sigma_t_fn = compute_sigma_t_fn

        def compute_mu_t(
            self,
            t,
            x0,
            x1,
        ):
            """"""  # noqa

            t = broadcast_to_target_shape(t, x0.shape)
            self._verify_shapes(x0, x1)
            self._verify_shapes(t, x1)
            return self._compute_mu_t_fn(t, x0, x1)

        def compute_sigma_t(
            self,
            t,
        ):
            """"""  # noqa

            if require_prng:
                return self._compute_sigma_t_fn(t) * self._sigma
            return zeros_like(t)

        def compute_ut(
            self,
            t,
            xt,
            x0,
            x1,
        ):
            """"""  # noqa

            t = broadcast_to_target_shape(t, x0.shape)
            self._verify_shapes(xt, x1)
            self._verify_shapes(x0, x1)
            self._verify_shapes(t, x1)
            return self._compute_ut_fn(t, xt, x0, x1)

    return CustomProbabilityPath(sigma, prng=prng)
