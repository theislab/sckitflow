from sc_flow._runtime import (
    raise_runtime_error_on_backend_failed_import,
    raise_runtime_error_on_backend_not_supported,
    set_jax_import_failed,
    set_torch_import_failed,
)


def get_dummy_network(input_dim, output_dim, hidden_dims=(None,), sigma=0.5):
    # set_backend(backend)
    from sc_flow._runtime import BACKEND

    print(BACKEND)
    if BACKEND == "torch":
        try:
            from sc_flow.core.nn._modules import MLP
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
        from torch import Tensor, rand
        from torch.nn import Module
        from torch.nn.functional import mse_loss

        from sc_flow.core.nn._modules import MLP
        from sc_flow.flow.probability_paths import LinearGaussianProbabilityPath

        class MethodClassTorch(Module):
            def __init__(self, network, prob_path, time_sampler) -> None:
                super().__init__()
                self.network = network
                self.prob_path = prob_path
                self.time_sampler = time_sampler
                self.train_called = False
                self.eval_called = 0

            def train_step(self, batch, prng_step_fn=None) -> Tensor:
                target = batch["target"]
                source = batch["source"]
                batch_size = target.shape[0]
                t = self.time_sampler(batch_size, device=target.device)
                xt = self.prob_path.compute_xt(t, source, target)
                vt = self.network.forward(xt)
                ut = self.prob_path.compute_ut(t, xt, source, target)
                loss = mse_loss(vt, ut)
                self.train_called = True
                return loss

            def validation_step(self, batch, prng_step_fn=None) -> None:
                self.eval_called = 1
                pass

        network = MLP(input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims)
        prob_path = LinearGaussianProbabilityPath(
            sigma=sigma,
        )
        time_sampler = rand
        method = MethodClassTorch(network=network, prob_path=prob_path, time_sampler=time_sampler)

        return method

    elif BACKEND == "jax":
        import jax.numpy as jnp
        from flax import linen as nn
        from jax import Array, random

        from sc_flow.backends.jax.nn._modules import MLP
        from sc_flow.backends.jax.probability_paths._probability_paths import LinearGaussianProbabilityPath

        class MethodClassJAX(nn.Module):
            def __init__(
                self,
                network,
                prob_path,
                time_sampler,
                input_dim,
                prng,
            ) -> None:
                super().__init__()
                self.network = network
                self.prob_path = prob_path
                self.time_sampler = time_sampler
                self.train_called = False
                self.eval_called = 0
                self._prng = prng

            def train_step(self, batch, prng_step_fn=None) -> Array:
                _, prng_step_fn = random.split(self._prng, 2)

                target = batch["target"]
                source = batch["source"]
                batch_size = target.shape[0]
                t = self.time_sampler(prng_step_fn, (batch_size, 1))
                xt = self.prob_path.compute_xt(t, source, target, prng=prng_step_fn)
                params = self.network.init(prng_step_fn, jnp.zeros((batch_size, self.network.input_dim)))

                vt = self.network.apply(params, xt, train=False)
                ut = self.prob_path.compute_ut(t, xt, source, target)
                loss = jnp.mean((vt - ut) ** 2)
                self.train_called = True
                return loss

            def validation_step(self, batch, prng_step_fn=None) -> None:
                self.eval_called = 1
                pass

        rng_jax = random.PRNGKey(0)
        network = MLP(input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims)
        prob_path = LinearGaussianProbabilityPath(
            sigma=sigma,
        )
        time_sampler = random.normal
        method = MethodClassJAX(
            network=network, prob_path=prob_path, time_sampler=time_sampler, prng=rng_jax, input_dim=input_dim
        )

        return method
