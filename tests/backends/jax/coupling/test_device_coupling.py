"""``couple_device`` (the jitted, device-native OT coupling used every training step).

The solve + sample_joint run under ``jax.jit`` for speed (measured ~245x vs eager on GPU). These tests
pin the behaviour that matters: valid in-range indices, determinism in the PRNG key, and that jitting
did not change the math — the sampled plan matches an independent *eager* sinkhorn + sample_joint.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
import torch

pytest.importorskip("ott")

from sc_flow.backends.jax.coupling._device import couple_device, torch_to_jax


def _reps(n_src=64, n_tgt=48, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    src = torch.randn(n_src, d, generator=g)
    tgt = torch.randn(n_tgt, d, generator=g) + 0.5
    return src, tgt


def test_couple_device_returns_valid_indices():
    src, tgt = _reps()
    si, ti = couple_device(src, tgt, key=jax.random.PRNGKey(0))
    assert si.dtype == torch.long and ti.dtype == torch.long
    assert si.shape == ti.shape
    assert int(si.min()) >= 0 and int(si.max()) < src.shape[0]
    assert int(ti.min()) >= 0 and int(ti.max()) < tgt.shape[0]


def test_couple_device_deterministic_in_key():
    src, tgt = _reps()
    a = couple_device(src, tgt, key=jax.random.PRNGKey(7))
    b = couple_device(src, tgt, key=jax.random.PRNGKey(7))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    # A different key generally samples a different pairing.
    c = couple_device(src, tgt, key=jax.random.PRNGKey(8))
    assert not (torch.equal(a[0], c[0]) and torch.equal(a[1], c[1]))


def test_jitted_couple_matches_eager_sinkhorn():
    """The jitted coupler must sample the same joint as an independent eager sinkhorn solve."""
    from ott.geometry import costs, pointcloud
    from ott.problems.linear import linear_problem
    from ott.solvers.linear import sinkhorn
    from ott.solvers.utils import sample_joint

    src, tgt = _reps(seed=1)
    key = jax.random.PRNGKey(3)

    # eager reference (same config as couple_device's linear defaults)
    geom = pointcloud.PointCloud(torch_to_jax(src), torch_to_jax(tgt), cost_fn=costs.SqEuclidean(),
                                 epsilon=1.0, scale_cost="mean")
    tmat = sinkhorn.Sinkhorn(threshold=1e-3)(linear_problem.LinearProblem(geom)).matrix
    esi, eti = sample_joint(key, tmat)

    si, ti = couple_device(src, tgt, key=key)
    np.testing.assert_array_equal(np.asarray(si.cpu()), np.asarray(esi))
    np.testing.assert_array_equal(np.asarray(ti.cpu()), np.asarray(eti))
