from typing import Tuple, Union

import jax
import jax.numpy as jnp

from jaxrl2.data.skill_discovery.buffers import NstepSkillBatch, SkillBatch
from jaxrl.networks.common import InfoDict, Model, Params, PRNGKey
from jaxrl2.networks.ensemble_model import EnsembleState, DeterministicEnsemble
from jaxrl2.agents.skillsac.utils import get_unsupervised_discovery_reward


def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_util.tree_map(
        lambda x, y: x * (1 - tau) + y * tau, target_critic.params,
        critic.params)

    return target_critic.replace(params=new_target_params)


def update_skill_critic(key: PRNGKey,
                        skill_actor: Model,
                        skill_critic: Model,
                        target_skill_critic: Model,
                        skill_temp: Model,
                        dyn_entropy_temp: Model,
                        ens: DeterministicEnsemble,
                        ens_state: EnsembleState,
                        batch: Union[NstepSkillBatch, SkillBatch],
                        discount: float,
                        backup_entropy: bool) -> Tuple[Model, EnsembleState, InfoDict]:
    if isinstance(batch, NstepSkillBatch):
        next_obs = batch.n_step_next_observations
        rew = batch.n_step_rewards
        mask = batch.n_step_masks
        # next_skill = batch.n_nstep_next_skills
    else:
        next_obs = batch.next_observations
        rew = batch.rewards
        mask = batch.masks
        # next_skill = batch.next_skills

    dist = skill_actor(next_obs)
    next_skills = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_skills)

    # get info gain
    info_gain, new_ens_state = ens.get_info_gain(
        input=jnp.concatenate([next_obs, next_skills], axis=-1),
        state=ens_state, update_normalizer=False)

    next_q1, next_q2 = target_skill_critic(next_obs, next_skills)
    next_q = jnp.minimum(next_q1, next_q2)

    target_q = rew + discount * mask * next_q

    if backup_entropy:
        dyn_ent_coef, _ = dyn_entropy_temp()
        act_ent_coef, _ = skill_temp()
        total_entropy = dyn_ent_coef * info_gain - act_ent_coef * next_log_probs
        target_q += discount * mask * total_entropy

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = skill_critic.apply_fn({'params': critic_params}, batch.observations,
                                       batch.skills)
        critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
        return critic_loss, {
            'skill_critic_loss': critic_loss,
            'skill_q1': q1.mean(),
            'skill_q2': q2.mean()
        }

    new_critic, info = skill_critic.apply_gradient(critic_loss_fn)

    return new_critic, new_ens_state, info


def update_critic(key: PRNGKey,
                  actor: Model,
                  critic: Model,
                  target_critic: Model,
                  temp: Model,
                  ens: DeterministicEnsemble,
                  ens_state: EnsembleState,
                  batch: Union[NstepSkillBatch, SkillBatch],
                  discount: float,
                  backup_entropy: bool,
                  num_skill_samples: int) -> Tuple[Model, InfoDict]:
    key, skill_key = jax.random.split(key, 2)
    # TODO: See if the current skill or next skill shall be used here
    # get the critic value for the next state and the current skill V(s_t+1, z_t)
    actor_inp = jnp.concatenate([batch.next_observations, batch.skills], axis=-1)
    dist = actor(actor_inp)
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)
    next_q1, next_q2 = target_critic(actor_inp, next_actions)
    next_q = jnp.minimum(next_q1, next_q2)

    # get skill discovery reward
    skill_rew = lambda obs, skill, rng: get_unsupervised_discovery_reward(
        observation=obs,
        skill=skill,
        ensemble=ens,
        ens_state=ens_state,
        rng=rng,
        skill_sampling_dist='Gaussian',
        num_skill_samples=num_skill_samples,
        squash_skill=True,
    )
    skill_key = jax.random.split(skill_key, actor_inp.shape[0])

    # get skill reward
    skill_rewards = jax.vmap(skill_rew)(batch.observations, batch.skills, skill_key)

    target_q = skill_rewards + discount * batch.masks * next_q

    if backup_entropy:
        ent_coef, _ = temp()
        target_q -= discount * batch.masks * ent_coef * next_log_probs

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        critic_inp = jnp.concatenate([batch.observations, batch.skills], axis=-1)
        q1, q2 = critic.apply_fn({'params': critic_params}, critic_inp,
                                 batch.actions)
        critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean(),
            'q2': q2.mean(),
            'skill_rewards': skill_rewards.mean(),
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info
