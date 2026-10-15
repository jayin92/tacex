from typing import Tuple, Union

import jax.numpy as jnp
import jax.random

from jaxrl2.data.skill_discovery.buffers import NstepSkillBatch, SkillBatch
from jaxrl.networks.common import InfoDict, Model, Params, PRNGKey
from jaxrl2.networks.ensemble_model import EnsembleState, DeterministicEnsemble


def update_skill_actor(key: PRNGKey,
                       skill_actor: Model,
                       skill_critic: Model,
                       skill_temp: Model,
                       target_skill_actor: Model,
                       dyn_entropy_temp: Model,
                       ens: DeterministicEnsemble,
                       ens_state: EnsembleState,
                       batch: Union[NstepSkillBatch, SkillBatch]) -> Tuple[Model, EnsembleState, InfoDict]:
    key, target_key = jax.random.split(key, 2)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, Tuple[EnsembleState, InfoDict]]:
        dist = skill_actor.apply_fn({'params': actor_params}, batch.observations)
        skill = dist.sample(seed=key)
        log_probs = dist.log_prob(skill)
        q1, q2 = skill_critic(batch.observations, skill)
        q = jnp.minimum(q1, q2)

        # getting info gain objective
        target_skill = target_skill_actor(batch.observations).sample(seed=target_key)
        target_inp = jnp.concatenate([batch.observations, target_skill], axis=-1)
        inp = jnp.concatenate([batch.observations, skill], axis=-1)
        total_inp = jnp.concatenate([inp, target_inp], axis=0)
        info_gain, new_ens_state = ens.get_info_gain(input=total_inp,
                                                     state=ens_state,
                                                     update_normalizer=True)
        info_gain, target_info_gain = info_gain[:skill.shape[0]], info_gain[skill.shape[0]:]
        dyn_ent_coef, _ = dyn_entropy_temp()
        act_ent_coef, _ = skill_temp()
        total_entropy = dyn_ent_coef * info_gain - act_ent_coef * log_probs
        actor_loss = -(total_entropy + q).mean()
        return actor_loss, (new_ens_state, {
            'skill_actor_loss': actor_loss,
            'skill_entropy': -log_probs.mean(),
            'info_gain': info_gain.mean(),
            'target_info_gain': target_info_gain.mean(),
            # 'actor_info_gain_nans': jnp.isnan(info_gain).sum(),
            # 'target_actor_info_gain_nans': jnp.isnan(target_info_gain).sum(),
        })

    new_actor, (new_ens_state, info) = skill_actor.apply_gradient(actor_loss_fn)

    return new_actor, new_ens_state, info


def update_actor(key: PRNGKey, actor: Model, critic: Model, temp: Model,
                 batch: Union[NstepSkillBatch, SkillBatch]) -> Tuple[Model, InfoDict]:
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        actor_state = jnp.concatenate([batch.observations, batch.skills], axis=-1)
        dist = actor.apply_fn({'params': actor_params}, actor_state)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        q1, q2 = critic(actor_state, actions)
        q = jnp.minimum(q1, q2)
        ent_coef, _ = temp()
        actor_loss = (log_probs * ent_coef - q).mean()
        return actor_loss, {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean()
        }

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info
