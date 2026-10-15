from typing import Tuple

import jax
import jax.numpy as jnp

from jaxrl.datasets import Batch
from jaxrl.networks.common import InfoDict, Model, Params, PRNGKey


def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_util.tree_map(
        lambda x, y: x * (1 - tau) + y * tau, target_critic.params,
        critic.params)

    return target_critic.replace(params=new_target_params)


def update(key: PRNGKey, actor: Model, critic: Model, target_critic: Model,
           temp: Model, batch: Batch, discount: float,
           backup_entropy: bool) -> Tuple[Model, InfoDict]:
    dist = actor(batch.next_observations)
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)
    next_q, _, _ = target_critic(batch.next_observations, next_actions)
    next_q = jnp.min(next_q, axis=0)

    target_q = batch.rewards + discount * batch.masks * next_q

    if backup_entropy:
        ent_coef, _ = temp()
        target_q -= discount * batch.masks * ent_coef * next_log_probs

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q, _, _ = critic.apply_fn({'params': critic_params}, batch.observations,
                                 batch.actions)
        q1, q2 = q[0], q[-1]
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean(),
            'q2': q2.mean()
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info
