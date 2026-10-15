from typing import Tuple

import jax
import jax.numpy as jnp

from jaxrl2.data import NstepBatch
from jaxrl.networks.common import InfoDict, Model, Params, PRNGKey
from jaxrl2.networks.ensemble_model import EnsembleState, DeterministicEnsemble


def update(key: PRNGKey,
           actor: Model,
           critic: Model,
           target_critic: Model,
           ens: DeterministicEnsemble,
           ens_state: EnsembleState,
           noise_std: float,
           noise_clip: float,
           batch: NstepBatch,
           discount: float,
           mask_expl_critic: bool,
           ) -> Tuple[Model, EnsembleState, InfoDict]:
    # check if both batches have the same observation
    batch_size = batch.next_observations.shape[0]

    # sample actions for the next observations
    key_n_step, key_one_step, key = jax.random.split(key, 3)

    # get n step action
    # s_{t+n} -> a_{t+n}
    next_actions_n_step = actor(batch.n_step_next_observations)
    noise = jax.random.normal(key=key_n_step, shape=next_actions_n_step.shape) * noise_std
    noise = jnp.clip(noise, min=-noise_clip, max=noise_clip)
    next_actions_n_step = jnp.clip(next_actions_n_step + noise, min=-1, max=1)

    # get one step action
    # s_{t+1} -> a_{t+1}, log_p(a_{t+1}|s_{t+1})
    next_actions_one_step = actor(batch.next_observations)
    noise = jax.random.normal(key=key_one_step, shape=next_actions_one_step.shape) * noise_std
    noise = jnp.clip(noise, min=-noise_clip, max=noise_clip)
    next_actions_one_step = jnp.clip(next_actions_one_step + noise, min=-1, max=1)

    # concatenate obs and actions
    # full_obs = [s_{t+n}, s_{t+1}, s_{t}]
    full_obs = jnp.concatenate([batch.n_step_next_observations, batch.next_observations,
                                batch.observations], axis=0)
    # full_act = [a_{t+n}, a_{t+1}, a_{t}]
    full_act = jnp.concatenate([next_actions_n_step, next_actions_one_step, batch.actions], axis=0)

    next_q, expl_next_q, state, obs = target_critic(full_obs, full_act)
    # next q for n step returns, q_{t+n}
    next_q = next_q[:, :batch_size]
    next_q = jnp.min(next_q, axis=0)

    # next q for 1 step exploration critic, q^{expl}_{t+1}
    expl_next_q = expl_next_q[:, batch_size: 2 * batch_size]
    expl_next_q = jnp.min(expl_next_q, axis=0)

    # extract next state z_{t+1}
    next_state = state[batch_size: 2 * batch_size]
    # extract state z_{t} and obs o_t
    state, obs = state[2 * batch_size:], obs[2 * batch_size:]

    # get info gain (r^{expl}_t)
    expl_reward, new_ens_state = ens.get_info_gain(
        input=jnp.concatenate([state, batch.actions], axis=-1),
        state=ens_state, update_normalizer=False)

    # Q(s_t, a_t) = \sum^{t+n}_{t} \gamma^i r_i + \gamma^{n} * q_{t+n}
    target_q = batch.n_step_rewards + discount * batch.n_step_masks * next_q

    # Q^{expl}(s_t, a_t) = r^{expl}_t + \gamma * q^{expl}_{t+1}
    if mask_expl_critic:
        target_expl_q = expl_reward + discount * batch.masks * expl_next_q
    else:
        target_expl_q = expl_reward + discount * expl_next_q

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # get q value
        q, qexpl, _, _ = critic.apply_fn({'params': critic_params}, batch.observations,
                                 batch.actions)

        critic_loss = ((q - target_q[jnp.newaxis]) ** 2).mean() + ((qexpl - target_expl_q[jnp.newaxis]) ** 2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q[0].mean(),
            'q2': q[1].mean(),
            'q1_expl': qexpl[0].mean(),
            'q2_expl': qexpl[1].mean(),
            'next_state': jax.lax.stop_gradient(next_state),
            'state': jax.lax.stop_gradient(state),
            'obs': jax.lax.stop_gradient(obs),
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, new_ens_state, info
