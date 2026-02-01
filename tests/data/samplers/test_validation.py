from sc_flow.data._composite import MatchedData
from sc_flow.data.samplers._validation import FValidationSampler

from ..shared import make_distribution, make_tree  # noqa


class TestValidationSampler:
    def test_properties_exposed(self):
        tree = make_tree()

        sampler = FValidationSampler(
            tree,
            lambda x: x,
            max_n_obs=4,
            n_nodes=2,
            replace_samples=True,
            replace_nodes=True,
            use_nodes_weights=False,
        )

        assert sampler.max_n_obs == 4
        assert sampler.n_nodes == 2
        assert sampler.replace_samples is True
        assert sampler.replace_nodes is True
        assert sampler.use_nodes_weights is False
        assert sampler.tree is tree

    def test_data_registered_on_init(self):
        tree = make_tree()

        sampler = FValidationSampler(
            tree,
            lambda x: x,
            max_n_obs=3,
            n_nodes=2,
        )

        data = sampler.data

        assert isinstance(data, tuple)
        assert len(data) == 2
        assert all(isinstance(b, MatchedData) for b in data)
        assert all(len(b.target) == 3 for b in data)

    def test_len_and_getitem(self):
        tree = make_tree()

        sampler = FValidationSampler(
            tree,
            lambda x: x,
            max_n_obs=2,
            n_nodes=2,
        )

        # __len__ returns number of nodes
        assert len(sampler) == 2

        # __getitem__ slicing
        slice_data = sampler[:1]
        assert isinstance(slice_data, tuple)
        assert len(slice_data) == 1

    def test_iteration_yields_batches(self):
        tree = make_tree()

        sampler = FValidationSampler(
            tree,
            lambda x: x,
            max_n_obs=2,
            n_nodes=2,
        )

        batches = list(iter(sampler))
        assert len(batches) == 2
        assert all(isinstance(b, MatchedData) for b in batches)
        assert all(len(b.target) == 2 for b in batches)

    def test_dispatch_fn_applied(self):
        tree = make_tree()

        sampler = FValidationSampler(
            tree,
            lambda x: ("processed", x),
            max_n_obs=1,
            n_nodes=1,
        )

        batch = sampler.data[0]
        assert batch[0] == "processed"
