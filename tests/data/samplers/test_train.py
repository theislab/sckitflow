from sckitflow.data._composite import MatchedData
from sckitflow.data.samplers._train import FTrainSampler

from ..shared import make_distribution, make_tree  # noqa


class TestTrainSampler:
    def test_properties_exposed(self):
        tree = make_tree()

        sampler = FTrainSampler(
            tree,
            lambda x: x,
            batch_size=7,
            n_nodes=2,
            replace_samples=True,
            replace_nodes=True,
            use_nodes_weights=False,
        )

        assert sampler.batch_size == 7
        assert sampler.n_nodes == 2
        assert sampler.replace_samples is True
        assert sampler.replace_nodes is True
        assert sampler.use_nodes_weights is False
        assert sampler.tree is tree

    def test_sample_uses_configured_sizes(self):
        tree = make_tree()

        sampler = FTrainSampler(
            tree,
            lambda x: x,
            batch_size=3,
            n_nodes=2,
            replace_nodes=True,
        )

        batches = sampler.sample()

        assert isinstance(batches, tuple)
        assert len(batches) == 2
        assert all(isinstance(b, MatchedData) for b in batches)
        assert all(len(b.target) == 3 for b in batches)

    def test_sample_with_source_none(self):
        tree = make_tree()

        sampler = FTrainSampler(
            tree,
            lambda x: x,
            batch_size=2,
            n_nodes=1,
        )

        batch = sampler.sample()[0]

        assert batch.source is None
        assert len(batch.target) == 2

    def test_dispatch_fn_applied(self):
        tree = make_tree()

        sampler = FTrainSampler(
            tree,
            lambda x: ("processed", x),
            batch_size=1,
            n_nodes=1,
        )

        batch = sampler.sample()[0]

        assert batch[0] == "processed"

    def test_node_sampling_respects_weights(self):
        tree = make_tree()

        sampler = FTrainSampler(
            tree,
            lambda x: x,
            batch_size=1,
            n_nodes=100,
            replace_nodes=True,
            use_nodes_weights=True,
        )

        batches = sampler.sample()
        targets = [len(b.target) for b in batches]

        assert set(targets).issubset({1})
