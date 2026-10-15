from typing import Tuple

import jax
import jax.numpy as jnp

from jaxrl.datasets import Batch
from jaxrl.networks.common import InfoDict, Model, Params, PRNGKey
from jaxrl2.networks.ensemble_model import EnsembleState, DeterministicEnsemble


def update(key: PRNGKey,
           actor: Model,
           critic: Model,
           target_critic: Model,
           temp: Model,
           dyn_entropy_temp: Model,
           ens: DeterministicEnsemble,
           ens_state: EnsembleState,
           batch: Batch,
           discount: float,
           backup_entropy: bool) -> Tuple[Model, EnsembleState, InfoDict]:
    batch_size = batch.next_observations.shape[0]
    dist = actor(batch.next_observations)
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)

    # extract next q values, state representations and obs representations from target critic
    full_obs = jnp.concatenate([batch.next_observations, batch.observations], axis=0)
    full_act = jnp.concatenate([next_actions, batch.actions])

    q, state, obs = target_critic(full_obs, full_act)
    next_q, next_state = q[:, :batch_size], state[:batch_size]
    state, obs = state[batch_size:], obs[batch_size:]
    # next_q, next_state, next_obs = target_critic(batch.next_observations, next_actions)
    next_q = jnp.min(next_q, axis=0)
    # actions are not used in getting state and observations
    # _, state, obs = target_critic(batch.observations, next_actions)

    # get info gain
    info_gain, new_ens_state = ens.get_info_gain(
        input=jnp.concatenate([next_state, next_actions], axis=-1),
        state=ens_state, update_normalizer=False)

    target_q = batch.rewards + discount * batch.masks * next_q

    if backup_entropy:
        dyn_ent_coef, _ = dyn_entropy_temp()
        act_ent_coef, _ = temp()
        total_entropy = dyn_ent_coef * info_gain - act_ent_coef * next_log_probs
        target_q += discount * batch.masks * total_entropy

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q, _, _ = critic.apply_fn({'params': critic_params}, batch.observations,
                                 batch.actions)
        critic_loss = ((q - target_q[jnp.newaxis]) ** 2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q[0].mean(),
            'q2': q[1].mean(),
            'next_state': jax.lax.stop_gradient(next_state),
            'state': jax.lax.stop_gradient(state),
            'obs': jax.lax.stop_gradient(obs),
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, new_ens_state, info
