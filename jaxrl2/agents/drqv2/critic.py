from typing import Tuple

import jax.numpy as jnp
import jax.random

from jaxrl2.data import NstepBatch
from jaxrl.networks.common import InfoDict, Model, Params


def update(key: jax.random.PRNGKey, actor: Model, critic: Model, target_critic: Model, batch: NstepBatch,
           noise_std: float, noise_clip: float,
           discount: float,
           ) -> Tuple[Model, InfoDict]:
    next_actions = actor(batch.n_step_next_observations)
    noise = jax.random.normal(key=key, shape=next_actions.shape) * noise_std
    noise = jnp.clip(noise, min=-noise_clip, max=noise_clip)
    next_actions = jnp.clip(next_actions + noise, min=-1, max=1)
    next_q1, next_q2 = target_critic(batch.n_step_next_observations, next_actions)
    next_q = jnp.minimum(next_q1, next_q2)

    target_q = batch.n_step_rewards + discount * batch.n_step_masks * next_q

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = critic.apply_fn({'params': critic_params}, batch.observations,
                                 batch.actions)
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean(),
            'q2': q2.mean()
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info
