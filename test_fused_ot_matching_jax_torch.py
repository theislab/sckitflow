from functools import partial

import jax.numpy as jnp
import numpy as np
import ot as pot
import scipy as sp
from ott.solvers.utils import match_quadratic

ot_fn = partial(pot.gromov.entropic_fused_gromov_wasserstein, epsilon=1.0, alpha=1.0)

source = np.random.rand(1000, 50)
target = np.random.rand(1000, 50)
source_source = np.random.rand(1000, 50)
target_target = np.random.rand(1000, 50)

cost_fn = lambda source, target: sp.spatial.distance.cdist(source, target) ** 2

distance_matrix_xx = cost_fn(source_source, source_source)
distance_matrix_yy = cost_fn(target_target, target_target)
distance_matrix_xy = cost_fn(source, target)

coupling_matrix = ot_fn(
    distance_matrix_xy,
    distance_matrix_xx,
    distance_matrix_yy,
)


source_source_batch = jnp.array(source_source)
target_target_batch = jnp.array(target_target)
source_batch = jnp.array(source)
target_batch = jnp.array(target)

out = match_quadratic(xx=source_source_batch, yy=target_target_batch, x=source_batch, y=target_batch, fused_penalty=0.0)


# print(np.array(out.matrix) - coupling_matrix)

print(np.abs(np.array(out) - coupling_matrix).min())
print(np.abs(np.array(out) - coupling_matrix).max())
print(np.abs(np.array(out) - coupling_matrix).mean())

print(sp.stats.pearsonr(np.array(out).flatten(), coupling_matrix.flatten()))
print(sp.stats.spearmanr(np.array(out).flatten(), coupling_matrix.flatten()))
