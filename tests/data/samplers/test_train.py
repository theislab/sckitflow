from itertools import islice

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


class TestPerNodeIteration:
    """A node is the unit of training: iteration yields one node at a time."""

    def test_iteration_yields_single_nodes_not_rounds(self):
        tree = make_tree()
        sampler = FTrainSampler(tree, lambda x: x, batch_size=3, n_nodes=2, replace_nodes=True)

        nodes = list(islice(sampler, 5))

        assert len(nodes) == 5
        assert all(isinstance(node, MatchedData) for node in nodes)
        assert all(len(node.target) == 3 for node in nodes)

    def test_the_stream_is_unbounded_by_default(self):
        tree = make_tree()
        sampler = FTrainSampler(tree, lambda x: x, batch_size=1, n_nodes=1, replace_nodes=True)

        assert sampler.max_iter_steps is None
        # Far more nodes than a single round holds: fresh rounds are drawn as needed.
        assert len(list(islice(sampler, 25))) == 25

    def test_max_iter_steps_bounds_the_stream(self):
        tree = make_tree()
        sampler = FTrainSampler(tree, lambda x: x, batch_size=1, n_nodes=2, replace_nodes=True, max_iter_steps=3)

        assert len(list(sampler)) == 3

    def test_max_iter_steps_can_stop_mid_round(self):
        """The bound counts nodes, so it need not fall on a round boundary."""
        tree = make_tree()
        sampler = FTrainSampler(tree, lambda x: x, batch_size=1, n_nodes=4, replace_nodes=True, max_iter_steps=5)

        assert len(list(sampler)) == 5

    def test_the_stream_is_re_iterable(self):
        """Lightning builds a fresh iterator per epoch, so exhaustion must not be sticky."""
        tree = make_tree()
        sampler = FTrainSampler(tree, lambda x: x, batch_size=1, n_nodes=1, replace_nodes=True, max_iter_steps=2)

        assert len(list(sampler)) == 2
        assert len(list(sampler)) == 2

    def test_rounds_are_drawn_lazily(self):
        """A round is only drawn once the previous one is spent."""
        tree = make_tree()
        sampler = FTrainSampler(tree, lambda x: x, batch_size=1, n_nodes=3, replace_nodes=True)
        calls = []
        original_sample = sampler.sample
        sampler.sample = lambda: (calls.append(1), original_sample())[1]

        list(islice(sampler, 3))
        assert len(calls) == 1

        list(islice(sampler, 4))
        assert len(calls) == 3  # a fresh iterator, then a second round for the 4th node
