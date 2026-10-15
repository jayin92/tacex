from typing import Tuple

import jax.numpy as jnp
import jax.random

from jaxrl2.data import NstepBatch
from jaxrl.networks.common import InfoDict, Model, Params


def clip(x, min: float = -1.0, max: float = 1.0):
    # use custom clipping to allow gradient flow through the clip function
    clipped_x = jax.lax.stop_gradient(jnp.clip(x, min=min, max=max))
    x = x - jax.lax.stop_gradient(x) + clipped_x
    return x


def update(actor: Model,
           critic: Model,
           batch: NstepBatch,
           noise_std: float,
           noise_clip: float,
           key: jax.random.PRNGKey,
           ) -> Tuple[Model, InfoDict]:
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        actions = actor.apply_fn({'params': actor_params}, batch.observations)
        noise = jax.random.normal(key=key, shape=actions.shape) * noise_std
        noise = jnp.clip(noise, min=-noise_clip, max=noise_clip)
        actions = clip(actions + noise, min=-1, max=1)
        q1, q2 = critic(batch.observations, actions)
        q = jnp.minimum(q1, q2)
        actor_loss = -q.mean()
        return actor_loss, {'actor_loss': actor_loss}

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info
