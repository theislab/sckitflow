from collections.abc import Callable

import pytest

from sc_flow._utils import (
    get_fn_args_names_and_types,
    get_fn_kwargs_names_and_types,
    verify_fn_args,
)

from sc_flow._runtime import (
    raise_runtime_error_on_backend_failed_import,
    raise_runtime_error_on_backend_not_supported,
    set_jax_import_failed,
    set_torch_import_failed,
    BACKEND
)

Tfn1 = {"template": Callable[[float, float, float], float], "is_valid": True}
Tfn2 = {"template": Callable[[float, float], float], "is_valid": False}
Tfn3 = {"template": Callable[[float, int, float], float], "is_valid": False}


def fn_with_args(
    t: float,
    x0: float,
    x1: float,
) -> float:
    return (1 - t) * x1 - t * x0


def fn_with_kwargs(
    t: float,
    x0: float,
    x1: float,
    kwarg1: int = 0,
    kwarg2: float = 0.0,
) -> float:
    return (1 - t) * x1 - t * x0


class TestUtils:
    @pytest.mark.parametrize("fn", [fn_with_args, fn_with_kwargs])
    def test_get_fn_args_names_and_types(
        self,
        fn: dict[str, dict[str, type]],
    ) -> None:
        arg_types_dict = get_fn_args_names_and_types(fn)
        assert arg_types_dict == {"t": float, "x0": float, "x1": float}

    @pytest.mark.parametrize("fn", [fn_with_args, fn_with_kwargs])
    def test_get_fn_kwargs_names_and_types(
        self,
        fn: dict[str, Callable],
    ) -> None:
        if fn.__name__ == "fn_with_args":
            expected_kwargs = {}
        elif fn.__name__ == "fn_with_kwargs":
            expected_kwargs = {"kwarg1": int, "kwarg2": float}
        arg_types_dict = get_fn_kwargs_names_and_types(fn)
        assert arg_types_dict == expected_kwargs

    @pytest.mark.parametrize("Tfn_dict", [Tfn1, Tfn2, Tfn3])
    def test_verify_fn_args(
        self,
        Tfn_dict: dict[int, type[Callable]] | int,
    ) -> None:
        if not Tfn_dict["is_valid"]:
            with pytest.raises(
                TypeError,
                match=r"",
            ):
                verify_fn_args(fn_with_args, Tfn_dict["template"])
            return None
        else:
            verify_fn_args(fn_with_args, Tfn_dict["template"])


def get_dummy_network(input_dim,
                      output_dim,
                      hidden_dims=None,
                      sigma=0.5,
                      ):
    if BACKEND == "torch":
        try:
            from sc_flow.backends.torch.nn._modules import MLP
        except (ImportError, ModuleNotFoundError):
            set_torch_import_failed(True)
            raise_runtime_error_on_backend_failed_import()
    elif BACKEND == "jax":
        try:
            from sc_flow.backends.jax.nn._modules import MLP 
        except (ImportError, ModuleNotFoundError):
            set_jax_import_failed(True)
            raise_runtime_error_on_backend_failed_import()
    else:
        raise_runtime_error_on_backend_not_supported(BACKEND)
    
    

    if BACKEND == "torch":
        from sc_flow.backends.torch.nn._modules import MLP 
        from sc_flow.backends.torch.probability_paths import LinearGaussianProbabilityPath
        from torch.nn.functional import mse_loss
        from torch import rand
        from torch.nn import Module

        class MethodClassTorch(Module):
            def __init__(self,
                        network,
                        prob_path,
                        time_sampler) -> None:
                
                super().__init__()
                self.network = network
                self.prob_path = prob_path
                self.time_sampler = time_sampler
                self.train_called = False
                self.eval_called = 0

            def train_step(self,
                        batch,
                        prng_step_fn=None) -> None:
                
                target = batch["target"]
                source = batch["source"]
                batch_size = target.shape[0]
                t = self.time_sampler(batch_size, device=target.device)
                xt = self.prob_path.compute_xt(t, source, target)
                vt = self.network.forward(xt)
                ut = self.prob_path.compute_ut(t,xt, source, target)
                loss =  mse_loss(vt, ut)
                self.train_called = True
                return loss
            
            def validation_step(self,
                                batch,
                                prng_step_fn=None) -> None:
                self.eval_called = 1
                pass


        network = MLP(input_dim=input_dim,
                      output_dim=output_dim,
                      hidden_dims=hidden_dims)
        network._make_modules()
        prob_path = LinearGaussianProbabilityPath(sigma=sigma, )
        time_sampler = rand
        method = MethodClassTorch(network=network, prob_path=prob_path, time_sampler=time_sampler)


        return method
    
    elif BACKEND == "jax":
        pass
        """
        from sc_flow.backends.jax.nn._modules import MLP 
        network = MLP(input_dim=input_dim,
                      output_dim=output_dim,
                      hidden_dims=hidden_dims)
        return network
        """