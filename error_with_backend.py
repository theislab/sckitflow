from tests.utils import get_dummy_network

from sc_flow._runtime import (
    BACKEND,
    set_backend,
)
from sc_flow.trainer._trainer import FlowTrainer

print(BACKEND)

set_backend("jax")

print(BACKEND)

from sc_flow._runtime import BACKEND  # noqa

print(BACKEND)


input_dim = 10
output_dim = 10
hidden_dims = (20, 20)
batch_size = 32


class DummyValLoaderJax:
    """"""  # noqa

    def __init__(self):
        self.sample_calls = 0

    def sample(self, prng):
        """
        Docstring for sample

        :param self: Description
        :param prng: Description
        """
        from jax import random

        _, prng_step_fn_source, prng_step_fn_target = random.split(prng, 3)
        self.sample_calls += 1
        return {
            "condA": {
                "source": random.normal(prng_step_fn_source, (batch_size, output_dim)),
                "target": random.normal(prng_step_fn_target, (batch_size, output_dim)),
            }
        }


class DummyTrainLoaderJax:
    """Docstring for DummyTrainLoaderJax"""

    def __init__(self):
        """
        Docstring for __init__

        :param self: Description
        """
        self.sample_calls = 0

    def sample(self, prng):
        """
        Docstring for sample

        :param self: Description
        :param prng: Description
        """
        from jax import random

        _, prng_step_fn_source, prng_step_fn_target = random.split(prng, 3)
        self.sample_calls += 1
        return {
            "source": random.normal(prng_step_fn_source, (batch_size, output_dim)),
            "target": random.normal(prng_step_fn_target, (batch_size, output_dim)),
        }


class DummyCallbacks:
    """"""  # noqa

    def __init__(self):
        self.called_with = []

    def run_on_valid_step(self, validation_dict, condition):
        """
        Docstring for run_on_valid_step

        :param self: Description
        :param validation_dict: Description
        :param condition: Description
        """
        self.called_with.append((validation_dict, condition))
        return {"val_loss": 0.456}


method = get_dummy_network(input_dim, output_dim, hidden_dims, backend=BACKEND)
print(method)
trainloader = DummyTrainLoaderJax()
validationloader = DummyValLoaderJax()
trainer = FlowTrainer(method=method, require_prng=True, callbacks=DummyCallbacks())
trainer.fit(num_iterations=3, valid_freq=1, train_dataloader=trainloader, validation_dataloader=validationloader)
