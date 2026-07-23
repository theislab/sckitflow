from sc_flow.data._composite import MatchedData, NestedData
from sc_flow.data.samplers._train import FTrainSampler, MultiTransitionSampler

from ..shared import make_distribution, make_multi_transition_tree, make_tree  # noqa


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


class TestMultiTransitionSampler:
    def test_sample_returns_all_nodes(self):
        tree = make_multi_transition_tree(n_transitions=3, n_obs=10)

        sampler = MultiTransitionSampler(
            tree,
            batch_size=4,
            replace_samples=True,
        )

        batches = sampler.sample()

        assert isinstance(batches, tuple)
        assert len(batches) == len(tree.flatten())

    def test_n_transitions_equals_leaf_count(self):
        tree = make_multi_transition_tree(n_transitions=4, n_obs=10)

        sampler = MultiTransitionSampler(
            tree,
            batch_size=3,
            replace_samples=True,
        )

        assert sampler.n_transitions == len(tree.flatten())
        assert sampler.n_transitions == 4

    def test_dispatch_node_identity(self):
        tree = make_multi_transition_tree(n_transitions=2, n_obs=10)

        sampler = MultiTransitionSampler(
            tree,
            batch_size=3,
            replace_samples=True,
        )

        node = tree.flatten()[0]
        assert sampler._dispatch_node(node) is node

    def test_sample_batch_size_respected(self):
        tree = make_multi_transition_tree(n_transitions=3, n_obs=10)

        sampler = MultiTransitionSampler(
            tree,
            batch_size=5,
            replace_samples=True,
        )

        batches = sampler.sample()

        assert all(len(b.target) == 5 for b in batches)

    def test_sample_preserves_insertion_order(self):
        # Build tree with different-sized leaves to distinguish order
        data_dict = {}
        sizes = [5, 8, 12]
        for i, n in enumerate(sizes):
            target = make_distribution(n)
            source = make_distribution(n)
            data_dict[(str(i),)] = MatchedData(
                target_distribution=target, source_distribution=source
            )
        tree = NestedData(data_dict)

        sampler = MultiTransitionSampler(
            tree,
            batch_size=3,
            replace_samples=True,
        )

        batches = sampler.sample()

        # All batches have the same batch_size, so check count matches leaf count
        assert len(batches) == len(sizes)
        # Verify the source sizes match insertion order
        for batch, expected_source_size in zip(batches, sizes):
            assert batch.source is not None

    def test_sample_with_source_distributions(self):
        tree = make_multi_transition_tree(n_transitions=3, n_obs=10)

        sampler = MultiTransitionSampler(
            tree,
            batch_size=4,
            replace_samples=True,
        )

        batches = sampler.sample()

        assert all(b.source is not None for b in batches)
