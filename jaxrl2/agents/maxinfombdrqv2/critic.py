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
            'expl_reward': jax.lax.stop_gradient(expl_reward),
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, new_ens_state, info


def sample_next_state(key: PRNGKey,
                      ens: DeterministicEnsemble,
                      ens_state: EnsembleState,
                      state: jnp.ndarray,
                      actions: jnp.ndarray, predict_diff: bool,
                      sample_model: bool):
    input = jnp.concatenate([state, actions], axis=-1)
    ens_mean, ens_std = ens(input=input, state=ens_state, denormalize_output=True)
    noise_key, key = jax.random.split(key, 2)
    if sample_model:
        ens_mean = jax.random.choice(key=key, a=ens_mean, axis=0)
        ens_std = jax.random.choice(key=key, a=ens_std, axis=0)
    else:
        ens_mean = jnp.mean(ens_mean, axis=0)
        ens_std = jnp.mean(ens_std, axis=0)

    ens_mean = ens_mean[..., :state.shape[-1]]
    ens_std = ens_std[..., :state.shape[-1]]
    next_state = ens_mean + jax.random.normal(noise_key, shape=ens_std.shape) * ens_std
    if predict_diff:
        next_state = next_state + state
    return next_state


def update_with_imagined_batch(
        key: PRNGKey,
        actor: Model,
        critic: Model,
        target_critic: Model,
        ens: DeterministicEnsemble,
        ens_state: EnsembleState,
        state: jax.Array,
        expl_rew: jax.Array,
        noise_std: float,
        noise_clip: float,
        batch: NstepBatch,
        predict_diff: bool,
        sample_model: bool,
        discount: float,
        mask_expl_critic: bool):
    key, nex_state_key = jax.random.split(key, 2)
    next_state = sample_next_state(
        key=key,
        ens=ens,
        ens_state=ens_state,
        state=state,
        actions=batch.actions,
        predict_diff=predict_diff,
        sample_model=sample_model,
    )
    # observations are ignored if state is passed
    next_action = actor(batch.next_observations, state=next_state)
    noise = jax.random.normal(key=key, shape=next_action.shape) * noise_std
    noise = jnp.clip(noise, min=-noise_clip, max=noise_clip)
    next_action = jnp.clip(next_action + noise, min=-1, max=1)
    next_q, expl_next_q, _, _ = target_critic(batch.next_observations, next_action, state=next_state)
    next_q, expl_next_q = jnp.min(next_q, axis=0), jnp.min(expl_next_q, axis=0)

    # Q(s_t, a_t) = r_t + \gamma * q_{t+1}
    target_q = batch.rewards + discount * batch.masks * next_q

    # Q^{expl}(s_t, a_t) = r^{expl}_t + \gamma * q^{expl}_{t+1}
    if mask_expl_critic:
        target_expl_q = expl_rew + discount * batch.masks * expl_next_q
    else:
        target_expl_q = expl_rew + discount * expl_next_q

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # get q value
        q, qexpl, _, _ = critic.apply_fn({'params': critic_params}, batch.observations,
                                         batch.actions, state=state)

        critic_loss = ((q - target_q[jnp.newaxis]) ** 2).mean() + ((qexpl - target_expl_q[jnp.newaxis]) ** 2).mean()
        return critic_loss, {
            'imagined_critic_loss': critic_loss,
            'imagined_q1': q[0].mean(),
            'imagined_q2': q[1].mean(),
            'imagined_q1_expl': qexpl[0].mean(),
            'imagined_q2_expl': qexpl[1].mean(),
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info