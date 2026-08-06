import numpy as np

from sckitflow.data._composite import DistributionData
from sckitflow.data.samplers import MSampler
from sckitflow.data.samplers._train import MTrainSampler, FTrainSampler
from sckitflow.data.samplers._validation import MValidationSampler

from ..shared import make_distribution, make_tree


def _make_step_data(node):
    """Mimics extract_step_data: turns a MatchedData dict into a StepData-like dict."""
    target = node["target"]
    source = node.get("source")
    return {
        "target_state": target.state_data.X if target is not None else None,
        "source_state": source.state_data.X if source is not None else None,
    }


class TestMSampler:
    def test_properties_exposed(self):
        tree = make_tree()
        sampler = MSampler(
            tree,
            dispatch_fn=lambda x: x,
            replace_samples=True,
        )
        assert sampler.tree is tree
        assert sampler.replace_samples is True

    def test_flattened_data(self):
        tree = make_tree()
        sampler = MSampler(tree, dispatch_fn=lambda x: x)
        flattened = sampler.flattened_data

        assert isinstance(flattened, tuple)
        assert all(isinstance(n, dict) for n in flattened)

    def test_sample_returns_tuple_of_list(self):
        tree = make_tree()
        sampler = MSampler(tree, dispatch_fn=lambda x: x)
        result = sampler._sample(batch_size=2)

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], list)

    def test_sample_covers_all_nodes(self):
        tree = make_tree()
        sampler = MSampler(tree, dispatch_fn=lambda x: x)
        result = sampler._sample(batch_size=2)
        node_list = result[0]

        n_nodes = len(tree.flatten())
        assert len(node_list) == n_nodes

    def test_sample_batch_size_respected(self):
        tree = make_tree()
        sampler = MSampler(tree, dispatch_fn=lambda x: x)
        result = sampler._sample(batch_size=3)
        node_list = result[0]

        for node in node_list:
            assert len(node["target"]) == 3

    def test_dispatch_fn_applied(self):
        tree = make_tree()
        sampler = MSampler(tree, dispatch_fn=lambda x: ("processed", x))
        result = sampler._sample(batch_size=2)
        node_list = result[0]

        for item in node_list:
            assert item[0] == "processed"

    def test_dispatch_fn_applied_as_keyword(self):
        tree = make_tree()
        sampler = MSampler(
            tree,
            dispatch_fn=lambda node: {"dispatched": node},
        )
        result = sampler._sample(batch_size=2)
        node_list = result[0]

        for item in node_list:
            assert set(item) == {"dispatched"}

    def test_sample_from_distribution(self):
        tree = make_tree()
        distr = make_distribution(20)
        sampler = MSampler(tree, dispatch_fn=lambda x: x)
        sampled = sampler._sample_from_distr(distr, batch_size=4)

        assert isinstance(sampled, DistributionData)
        assert len(sampled) == 4


class TestMTrainSampler:
    def test_properties_exposed(self):
        tree = make_tree()
        sampler = MTrainSampler(
            tree,
            lambda x: x,
            batch_size=7,
            replace_samples=True,
        )

        assert sampler.batch_size == 7
        assert sampler.replace_samples is True
        assert sampler.tree is tree

    def test_sample_returns_tuple_of_list(self):
        tree = make_tree()
        sampler = MTrainSampler(
            tree,
            lambda x: x,
            batch_size=3,
        )

        batches = sampler.sample()

        assert isinstance(batches, tuple)
        assert len(batches) == 1
        assert isinstance(batches[0], list)

    def test_sample_covers_all_nodes(self):
        tree = make_tree()
        sampler = MTrainSampler(
            tree,
            lambda x: x,
            batch_size=2,
        )

        batches = sampler.sample()
        node_list = batches[0]
        n_nodes = len(tree.flatten())
        assert len(node_list) == n_nodes

    def test_sample_batch_size_respected(self):
        tree = make_tree()
        sampler = MTrainSampler(
            tree,
            lambda x: x,
            batch_size=3,
        )

        batches = sampler.sample()
        node_list = batches[0]

        for node in node_list:
            assert len(node["target"]) == 3

    def test_sample_with_source_none(self):
        tree = make_tree()
        sampler = MTrainSampler(
            tree,
            lambda x: x,
            batch_size=2,
        )

        batches = sampler.sample()
        node_list = batches[0]

        for node in node_list:
            assert node.get("source") is None
            assert len(node["target"]) == 2

    def test_dispatch_fn_applied(self):
        tree = make_tree()
        sampler = MTrainSampler(
            tree,
            lambda x: ("processed", x),
            batch_size=1,
        )

        batches = sampler.sample()
        node_list = batches[0]

        for item in node_list:
            assert item[0] == "processed"

    def test_dispatch_fn_applied_as_keyword(self):
        tree = make_tree()
        sampler = MTrainSampler(
            tree,
            dispatch_fn=lambda node: {"dispatched": node},
            batch_size=1,
        )

        batches = sampler.sample()
        node_list = batches[0]

        for item in node_list:
            assert set(item) == {"dispatched"}

    def test_iteration(self):
        tree = make_tree()
        sampler = MTrainSampler(
            tree,
            lambda x: x,
            batch_size=2,
            max_iter_steps=3,
        )

        results = list(sampler)
        assert len(results) == 3
        for batches in results:
            assert isinstance(batches, tuple)
            assert len(batches) == 1
            assert isinstance(batches[0], list)


class TestMValidationSampler:
    def test_properties_exposed(self):
        tree = make_tree()
        sampler = MValidationSampler(
            tree,
            lambda x: x,
            max_n_obs=4,
            replace_samples=True,
        )

        assert sampler.max_n_obs == 4
        assert sampler.replace_samples is True
        assert sampler.tree is tree

    def test_data_registered_on_init(self):
        tree = make_tree()
        sampler = MValidationSampler(
            tree,
            lambda x: x,
            max_n_obs=3,
        )

        data = sampler.data

        assert isinstance(data, tuple)
        assert len(data) == 1
        assert isinstance(data[0], list)

    def test_data_covers_all_nodes(self):
        tree = make_tree()
        sampler = MValidationSampler(
            tree,
            lambda x: x,
            max_n_obs=3,
        )

        data = sampler.data
        node_list = data[0]
        n_nodes = len(tree.flatten())
        assert len(node_list) == n_nodes

    def test_max_n_obs_respected(self):
        tree = make_tree()
        sampler = MValidationSampler(
            tree,
            lambda x: x,
            max_n_obs=3,
        )

        data = sampler.data
        node_list = data[0]

        for node in node_list:
            assert len(node["target"]) == 3

    def test_max_n_obs_larger_than_node_is_clamped(self):
        tree = make_tree()
        sampler = MValidationSampler(
            tree,
            lambda x: x,
            max_n_obs=1_000,
        )

        data = sampler.data
        node_list = data[0]

        for node in node_list:
            assert len(node["target"]) == 5

    def test_dispatch_fn_applied(self):
        tree = make_tree()
        sampler = MValidationSampler(
            tree,
            lambda x: ("processed", x),
            max_n_obs=1,
        )

        data = sampler.data
        node_list = data[0]

        for item in node_list:
            assert item[0] == "processed"

    def test_len_and_getitem(self):
        tree = make_tree()
        sampler = MValidationSampler(
            tree,
            lambda x: x,
            max_n_obs=2,
        )

        assert len(sampler) == 1

        slice_data = sampler[:1]
        assert isinstance(slice_data, tuple)
        assert len(slice_data) == 1

    def test_iteration_yields_list(self):
        tree = make_tree()
        sampler = MValidationSampler(
            tree,
            lambda x: x,
            max_n_obs=2,
        )

        batches = list(iter(sampler))
        assert len(batches) == 1
        assert isinstance(batches[0], list)


class TestMultiSamplerTrainStepIntegration:
    """Tests the flow from sampler.sample() through the trainer loop up to compute_loss.

    Verifies that MTrainSampler produces output that reaches compute_loss as
    list[StepData], while FTrainSampler produces StepData per call.
    """

    def test_msampler_delivers_list_to_train_step(self):
        """MTrainSampler.sample() yields a 1-element tuple; iterating gives a list.

        The trainer loop does `for step_data in batches:` — with MTrainSampler
        this iterates once and step_data is a list of dicts (one per node).
        """
        tree = make_tree()
        sampler = MTrainSampler(tree, _make_step_data, batch_size=2)

        batches = sampler.sample()

        # Trainer iterates over the tuple
        for step_data in batches:
            # With MTrainSampler, each element is a list of all nodes
            assert isinstance(step_data, list)
            assert len(step_data) == len(tree.flatten())
            for sd in step_data:
                assert "target_state" in sd
                assert isinstance(sd["target_state"], np.ndarray)

    def test_fsampler_delivers_single_stepdata_to_train_step(self):
        """FTrainSampler.sample() yields N-element tuple; iterating gives individual dicts."""
        tree = make_tree()
        sampler = FTrainSampler(
            tree, _make_step_data, batch_size=2, n_nodes=2, replace_nodes=True,
        )

        batches = sampler.sample()

        # Trainer iterates over the tuple
        for step_data in batches:
            # With FTrainSampler, each element is a single dict (StepData)
            assert isinstance(step_data, dict)
            assert "target_state" in step_data
            assert isinstance(step_data["target_state"], np.ndarray)

    def test_msampler_stepdata_shapes(self):
        """Each StepData in the list has the expected batch dimension."""
        tree = make_tree()
        batch_size = 3
        sampler = MTrainSampler(tree, _make_step_data, batch_size=batch_size)

        batches = sampler.sample()
        node_list = batches[0]

        for sd in node_list:
            assert sd["target_state"].shape[0] == batch_size

    def test_msampler_dispatch_fn_called_per_node(self):
        """dispatch_fn is called once per node, not once for the whole list."""
        call_count = 0

        def counting_dispatch(node):
            nonlocal call_count
            call_count += 1
            return _make_step_data(node)

        tree = make_tree()
        sampler = MTrainSampler(tree, counting_dispatch, batch_size=2)

        sampler.sample()

        n_nodes = len(tree.flatten())
        assert call_count == n_nodes

    def test_msampler_vs_fsampler_same_dispatch_output(self):
        """Both sampler types call the same dispatch_fn and produce dicts with the same keys."""
        tree = make_tree()

        m_sampler = MTrainSampler(tree, _make_step_data, batch_size=2)
        f_sampler = FTrainSampler(
            tree, _make_step_data, batch_size=2, n_nodes=2, replace_nodes=True,
        )

        m_batches = m_sampler.sample()
        f_batches = f_sampler.sample()

        # M: single list of dicts
        m_step_datas = m_batches[0]
        # F: tuple of dicts
        f_step_datas = list(f_batches)

        # Both produce dicts with the same keys
        m_keys = set(m_step_datas[0].keys())
        f_keys = set(f_step_datas[0].keys())
        assert m_keys == f_keys
