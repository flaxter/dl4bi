from functools import partial

import jax
import jax.numpy as jnp
from jax import jit
try:
    from sps.kernels import great_circle_dist, l2_dist
except ImportError:
    # sps (dl4bi-sps) not installed — provide pure-JAX fallbacks so the
    # rest of the package imports cleanly.  The geodesic Bias variants that
    # actually call great_circle_dist are not used by the coalescent tutorials.
    @jit
    def l2_dist(q: jax.Array, r: jax.Array) -> jax.Array:
        """Pairwise L2 distance: q [Q,D], r [K,D] -> [Q,K]."""
        return jnp.sqrt(jnp.sum((q[:, None, :] - r[None, :, :]) ** 2, axis=-1))

    @jit
    def great_circle_dist(q: jax.Array, r: jax.Array) -> jax.Array:
        """Pairwise great-circle distance (haversine): q,r [N,2] lat/lon radians -> [Q,K]."""
        lat1, lon1 = q[:, None, 0], q[:, None, 1]
        lat2, lon2 = r[None, :, 0], r[None, :, 1]
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = jnp.sin(dlat / 2) ** 2 + jnp.cos(lat1) * jnp.cos(lat2) * jnp.sin(dlon / 2) ** 2
        return 2 * jnp.arcsin(jnp.sqrt(a))


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
