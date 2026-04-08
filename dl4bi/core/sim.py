from functools import partial

import jax
import jax.numpy as jnp
from jax import jit
try:
    from sps.kernels import great_circle_dist, l2_dist
except ImportError:
    pass  # sps (dl4bi-sps) is optional; not used by the coalescent tutorials


@partial(jit, static_argnames=("causal",))
def delta_time(
    q: jax.Array,  # [Q, 1]
    r: jax.Array,  # [R, 1]
    causal: bool = True,
):
    d = r.T - q
    if causal:
        return jnp.where(d <= 0, d, jnp.inf)
    return d  # [Q, R]
