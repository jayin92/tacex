from typing import Tuple

import jax
import jax.numpy as jnp

from jaxrl.datasets import Batch
from jaxrl.networks.common import InfoDict, Model, Params, PRNGKey
from jaxrl2.networks.ensemble_model import EnsembleState, DeterministicEnsemble
from jaxrl2.utils.tactile.utils import TactileObs


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
    batch_size = batch.actions.shape[0]
    dist = actor(batch.next_observations)
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)

    # extract next q values, state representations and obs representations from target critic
    if batch.observations.tactile is not None:
        has_tactile_obs = True
        full_obs = jax.tree_util.tree_map(lambda x, y: jnp.concatenate([x, y], axis=0), batch.next_observations,
                                                                                        batch.observations)
    else:
        has_tactile_obs = False
        full_img = jnp.concatenate([batch.next_observations.image, batch.observations.image], axis=0)
        full_obs = TactileObs(image=full_img, tactile=None)

    full_act = jnp.concatenate([next_actions, batch.actions])

    q, state, embed = target_critic(full_obs, full_act)
    next_q, next_state = q[:, :batch_size], state[:batch_size]
    if has_tactile_obs:
        state, image_obs, tact_obs = state[batch_size:], embed.img_embed[batch_size:], embed.tact_embed[batch_size:]
    else:
        state, image_obs, tact_obs = state[batch_size:], embed.img_embed[batch_size:], None
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
            'image_obs': jax.lax.stop_gradient(image_obs),
            'tact_obs': jax.lax.stop_gradient(tact_obs),
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, new_ens_state, info
