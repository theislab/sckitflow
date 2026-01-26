from functools import partial

import jax.numpy as jnp
import numpy as np
import ot as pot
import scipy as sp
from ott.geometry import costs, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

ot_fn = partial(pot.sinkhorn, reg=5e-1)

source = np.random.rand(1000, 50)
target = np.random.rand(1000, 50)

src_weights = pot.unif(source.shape[0])
tgt_weights = pot.unif(target.shape[0])
cost_fn = lambda source, target: sp.spatial.distance.cdist(source, target) ** 2

distance_matrix = cost_fn(source, target)

distance_matrix = distance_matrix / distance_matrix.mean()

coupling_matrix = ot_fn(src_weights, tgt_weights, distance_matrix)


source_batch = jnp.array(source)
target_batch = jnp.array(target)

cost_fn_jax = costs.SqEuclidean()

threshold = 1e-3
tau_a = 1.0
tau_b = 1.0
geom = pointcloud.PointCloud(
    source_batch,
    target_batch,
    cost_fn=cost_fn_jax,
    epsilon=1.0,
    scale_cost="mean",
)
problem = linear_problem.LinearProblem(geom, tau_a=tau_a, tau_b=tau_b)
solver = sinkhorn.Sinkhorn(threshold=threshold)
out = solver(problem)


# print(np.array(out.matrix) - coupling_matrix)

print(np.abs(np.array(out.matrix) - coupling_matrix).min())
print(np.abs(np.array(out.matrix) - coupling_matrix).max())
print(np.abs(np.array(out.matrix) - coupling_matrix).mean())

print(sp.stats.pearsonr(np.array(out.matrix).flatten(), coupling_matrix.flatten()))
print(sp.stats.spearmanr(np.array(out.matrix).flatten(), coupling_matrix.flatten()))
