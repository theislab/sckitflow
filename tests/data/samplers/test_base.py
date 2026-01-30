import numpy as np
import pytest

from sc_flow.data._composite import DistributionData, MatchedData
from sc_flow.data.samplers import FSampler

from ..shared import make_distribution, make_tree  # noqa


class TestFSampler:
    def test_properties_exposed(self):
        tree = make_tree()
        sampler = FSampler(
            tree,
            dispatch_fn=lambda x: x,
            replace_samples=True,
            replace_nodes=True,
            use_nodes_weights=False,
        )
        assert sampler.tree is tree
        assert sampler.replace_samples is True
        assert sampler.replace_nodes is True
        assert sampler.use_nodes_weights is False

    def test_flattened_data_preprocess(self):
        tree = make_tree()
        sampler = FSampler(tree, dispatch_fn=lambda x: x, preprocess_fn=lambda x: x)
        flattened = sampler.flattened_data

        assert isinstance(flattened, np.ndarray)
        assert all(isinstance(n, MatchedData) for n in flattened)

    def test_groups_p_computation(self):
        tree = make_tree()
        sampler = FSampler(tree, dispatch_fn=lambda x: x)
        probs = sampler.groups_p

        assert isinstance(probs, np.ndarray)
        np.testing.assert_allclose(probs.sum(), 1.0)
        assert probs.shape[0] == len(tree.flatten())

    def test_sample_nodes(self):
        tree = make_tree()
        sampler = FSampler(tree, dispatch_fn=lambda x: x, replace_nodes=True)
        nodes = sampler._sample_nodes(2)

        assert len(nodes) == 2
        assert all(isinstance(n, MatchedData) for n in nodes)

    def test_sample_indices_without_replacement(self):
        tree = make_tree()
        sampler = FSampler(tree, dispatch_fn=lambda x: x, replace_samples=False)
        idxs = sampler._sample_indices(10, batch_size=5)

        assert len(idxs) == 5
        assert len(set(idxs)) == 5  # No duplicates

    def test_sample_indices_with_replacement(self):
        tree = make_tree()
        sampler = FSampler(tree, dispatch_fn=lambda x: x, replace_samples=True)
        idxs = sampler._sample_indices(3, batch_size=10)

        assert len(idxs) == 10
        assert idxs.max() < 3

    def test_sample_from_distribution(self):
        tree = make_tree()
        distr = make_distribution(20)
        sampler = FSampler(tree, dispatch_fn=lambda x: x)
        sampled = sampler._sample_from_distr(distr, batch_size=4)

        assert isinstance(sampled, DistributionData)
        assert len(sampled) == 4

    def test_sample_observations(self):
        tree = make_tree()
        sampler = FSampler(tree, dispatch_fn=lambda x: x)
        batch = sampler._sample_observations(tree.flatten()[0], batch_size=3)

        assert isinstance(batch, MatchedData)
        assert len(batch.target) == 3
        if batch.source is not None:
            assert len(batch.source) == 3

    def test_sample_returns_tuple(self):
        tree = make_tree()
        sampler = FSampler(tree, dispatch_fn=lambda x: x, replace_nodes=True)
        batches = sampler._sample(n_nodes=2, batch_size=2)

        assert isinstance(batches, tuple)
        assert len(batches) == 2
        assert all(isinstance(b, MatchedData) for b in batches)

    def test_dispatch_fn_applied(self):
        tree = make_tree()
        sampler = FSampler(tree, dispatch_fn=lambda x: ("processed", x))
        batch = sampler._sample_observations(tree.flatten()[0], batch_size=1)

        assert batch[0] == "processed"

    def test_error_when_sampling_too_many_without_replacement(self):
        tree = make_tree()
        sampler = FSampler(tree, dispatch_fn=lambda x: x, replace_samples=False)

        # Distribution with 2 items
        distr = make_distribution(2)

        # Sampling 5 without replacement should fail
        with pytest.raises(ValueError, match="Cannot take a larger sample than population"):
            sampler._sample_from_distr(distr, batch_size=5)

    def test_error_when_sampling_too_many_nodes_without_replacement(self):
        tree = make_tree()
        n_nodes_in_tree = len(tree.flatten())
        sampler = FSampler(tree, dispatch_fn=lambda x: x, replace_nodes=False)

        # Trying to sample more nodes than exist
        with pytest.raises(ValueError, match="Cannot take a larger sample than population"):
            sampler._sample_nodes(n_nodes_in_tree + 1)
