from sc_flow._runtime import (
    raise_runtime_error_on_backend_failed_import,
    raise_runtime_error_on_backend_not_supported,
    set_jax_import_failed,
    set_torch_import_failed,
    BACKEND
)

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