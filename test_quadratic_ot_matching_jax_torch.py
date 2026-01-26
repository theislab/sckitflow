from functools import partial

import jax.numpy as jnp
import numpy as np
import ot as pot
import scipy as sp
from ott.solvers.utils import match_quadratic

ot_fn = partial(pot.gromov.entropic_gromov_wasserstein, epsilon=1.0)

source = np.random.rand(1000, 50)
target = np.random.rand(1000, 50)

cost_fn = lambda source, target: sp.spatial.distance.cdist(source, target) ** 2

distance_matrix_xx = cost_fn(source, source)
distance_matrix_yy = cost_fn(target, target)

# distance_matrix_xx = distance_matrix_xx / distance_matrix_xx.mean()
# distance_matrix_xx = distance_matrix_yy / distance_matrix_yy.mean()

coupling_matrix = ot_fn(
    distance_matrix_xx,
    distance_matrix_yy,
)


source_batch = jnp.array(source)
target_batch = jnp.array(target)

out = match_quadratic(xx=source_batch, yy=target_batch)


# print(np.array(out.matrix) - coupling_matrix)

print(np.abs(np.array(out) - coupling_matrix).min())
print(np.abs(np.array(out) - coupling_matrix).max())
print(np.abs(np.array(out) - coupling_matrix).mean())

print(sp.stats.pearsonr(np.array(out).flatten(), coupling_matrix.flatten()))
print(sp.stats.spearmanr(np.array(out).flatten(), coupling_matrix.flatten()))
