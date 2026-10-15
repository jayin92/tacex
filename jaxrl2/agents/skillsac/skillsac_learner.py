"""Implementations of algorithms for continuous control."""

import functools
from typing import Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrl.agents.sac import temperature
from jaxrl2.agents.skillsac.actor import update_skill_actor, update_actor
from jaxrl2.agents.skillsac.critic import update_skill_critic, update_critic, target_update
from jaxrl2.data.skill_discovery.buffers import NstepSkillBatch, SkillBatch
from jaxrl.networks import critic_net, policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey

from jaxrl2.networks.ensemble_model import EnsembleState, DeterministicEnsemble, ProbabilisticEnsemble


@functools.partial(jax.jit,
                   static_argnames=('ens',
                                    'backup_entropy',
                                    'update_target',
                                    'use_log_transform',
                                    'predict_rewards',
                                    'predict_diff',
                                    'num_skill_samples',))
def _update_jit(
        rng: PRNGKey,
        actor: Model,
        skill_actor: Model,
        target_skill_actor: Model,
        critic: Model,
        target_critic: Model,
        skill_critic: Model,
        target_skill_critic: Model,
        temp: Model,
        skill_temp: Model,
        dyn_entropy_temp: Model,
        ens: DeterministicEnsemble,
        ens_state: EnsembleState,
        batch: Union[NstepSkillBatch, SkillBatch],
        discount: float,
        tau: float,
        target_entropy: float,
        target_skill_entropy: float,
        backup_entropy: bool,
        update_target: bool,
        use_log_transform: bool,
        predict_rewards: bool,
        predict_diff: bool,
        num_skill_samples: int,
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, Model, Model, Model, Model, Model, EnsembleState, InfoDict]:
    rng, key = jax.random.split(rng)

    skill_critic_key, critic_key = jax.random.split(key, 2)
    new_skill_critic, ens_state, skill_critic_info = update_skill_critic(
        key=skill_critic_key,
        skill_actor=skill_actor,
        skill_critic=skill_critic,
        target_skill_critic=target_skill_critic,
        skill_temp=skill_temp,
        dyn_entropy_temp=dyn_entropy_temp,
        ens=ens,
        ens_state=ens_state,
        batch=batch,
        discount=discount,
        backup_entropy=backup_entropy,
    )

    new_critic, critic_info = update_critic(
        key=critic_key,
        actor=actor,
        critic=critic,
        target_critic=target_critic,
        temp=temp,
        ens=ens,
        ens_state=ens_state,
        batch=batch,
        discount=discount,
        backup_entropy=backup_entropy,
        num_skill_samples=num_skill_samples)

    critic_info = skill_critic_info | critic_info

    if update_target:
        new_target_critic = target_update(new_critic, target_critic, tau)
        new_target_skill_critic = target_update(new_skill_critic, target_skill_critic, tau)
    else:
        new_target_critic = target_critic
        new_target_skill_critic = target_skill_critic

    rng, key = jax.random.split(rng)
    skill_actor_key, actor_key = jax.random.split(key, 2)
    new_skill_actor, ens_state, skill_actor_info = update_skill_actor(
        key=skill_actor_key,
        skill_actor=skill_actor,
        skill_critic=new_skill_critic,
        skill_temp=skill_temp,
        target_skill_actor=target_skill_actor,
        dyn_entropy_temp=dyn_entropy_temp,
        ens=ens,
        ens_state=ens_state,
        batch=batch,
    )

    new_actor, actor_info = update_actor(actor_key, actor, new_critic, temp, batch)

    new_skill_temp, skill_alpha_info = temperature.update(skill_temp, skill_actor_info['skill_entropy'],
                                                          target_skill_entropy, use_log_transform=use_log_transform)

    skill_alpha_info = {f'skill_actor_{key}': val for key, val in skill_alpha_info.items()}


    new_temp, alpha_info = temperature.update(temp, actor_info['entropy'],
                                              target_entropy, use_log_transform=use_log_transform)

    actor_info = actor_info | skill_actor_info

    if update_target:
        new_target_skill_actor = target_update(new_skill_actor, target_skill_actor, tau)
    else:
        new_target_skill_actor = target_skill_actor

    new_dyn_entropy_temp, dyn_ent_info = temperature.update(dyn_entropy_temp, skill_actor_info['info_gain'],
                                                            skill_actor_info['target_info_gain'],
                                                            use_log_transform=use_log_transform)
    dyn_ent_info = {f'dyn_ent_{key}': val for key, val in dyn_ent_info.items()}

    alpha_info = alpha_info | skill_alpha_info | dyn_ent_info

    if isinstance(batch, NstepSkillBatch):
        next_obs = batch.n_step_next_observations
    else:
        next_obs = batch.next_observations

    if predict_diff:
        outputs = next_obs - batch.observations
    else:
        outputs = next_obs
    if predict_rewards:
        outputs = jnp.concatenate([outputs, batch.rewards.reshape(-1, 1)], axis=-1)
    new_ens_state, (loss, mse) = ens.update(
        input=jnp.concatenate([batch.observations, batch.skills], axis=-1),
        output=outputs,
        state=ens_state,
    )
    ens_info = {'ens_nll': loss,
                'ens_mse': mse,
                'ens_inp_mean': ens_state.ensemble_normalizer_state.input_normalizer_state.mean.mean(),
                'ens_inp_std': ens_state.ensemble_normalizer_state.input_normalizer_state.std.mean(),
                # 'ens_inp_num_points': ens_state.ensemble_normalizer_state.input_normalizer_state.num_points,
                'ens_out_mean': ens_state.ensemble_normalizer_state.output_normalizer_state.mean.mean(),
                'ens_out_std': ens_state.ensemble_normalizer_state.output_normalizer_state.std.mean(),
                # 'ens_out_num_points': ens_state.ensemble_normalizer_state.output_normalizer_state.num_points,
                'ens_info_gain_mean': ens_state.ensemble_normalizer_state.info_gain_normalizer_state.mean.mean(),
                'ens_info_gain_std': ens_state.ensemble_normalizer_state.info_gain_normalizer_state.std.mean(),
                # 'ens_info_gain_num_points': ens_state.ensemble_normalizer_state.info_gain_normalizer_state.num_points,
                }

    return rng, \
        new_actor, \
        new_skill_actor, new_target_skill_actor, \
        new_critic, new_target_critic, \
        new_skill_critic, new_target_skill_critic, \
        new_temp, \
        new_skill_temp, \
        new_dyn_entropy_temp, \
        new_ens_state, \
        {
            **critic_info,
            **actor_info,
            **alpha_info,
            **ens_info
        }


class SkillSACLearner(object):

    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 skill_dim: int = 2,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 dyn_ent_lr: float = 3e-4,
                 dyn_wd: float = 0.0,
                 ens_lr: float = 3e-4,
                 ens_wd: float = 0.0,
                 hidden_dims: Sequence[int] = (256, 256),
                 model_hidden_dims: Sequence[int] = (256, 256),
                 num_heads: int = 5,
                 predict_reward: bool = True,
                 predict_diff: bool = True,
                 learn_std: bool = False,
                 discount: float = 0.99,
                 tau: float = 0.005,
                 target_update_period: int = 1,
                 target_entropy: Optional[float] = None,
                 target_skill_entropy: Optional[float] = None,
                 backup_entropy: bool = True,
                 init_temperature: float = 1.0,
                 init_temperature_dyn_entropy: float = 1.0,
                 init_mean: Optional[np.ndarray] = None,
                 use_log_transform: bool = True,
                 policy_final_fc_init_scale: float = 1.0,
                 num_skill_samples: int = 5,
                 ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """
        self.use_log_transform = use_log_transform
        action_dim = actions.shape[-1]
        self.skill_dim = skill_dim
        self.num_heads = num_heads
        self.predict_diff = predict_diff
        self.predict_rewards = predict_reward
        self.num_skill_samples = num_skill_samples

        if target_entropy is None:
            self.target_entropy = -action_dim
        else:
            self.target_entropy = target_entropy

        if target_skill_entropy is None:
            self.target_skill_entropy = -skill_dim
        else:
            self.target_skill_entropy = target_skill_entropy

        self.backup_entropy = backup_entropy

        self.tau = tau
        self.target_update_period = target_update_period
        self.discount = discount

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        # initialize skill and true actor
        skill_actor_key, actor_key = jax.random.split(actor_key, 2)

        actor_def = lambda dim: policies.NormalTanhPolicy(
            hidden_dims,
            dim,
            init_mean=init_mean,
            final_fc_init_scale=policy_final_fc_init_scale)

        # skill actor takes as input the observation and returns a skill
        skill_actor = Model.create(actor_def(skill_dim),
                                   inputs=[skill_actor_key, observations],
                                   tx=optax.adam(learning_rate=actor_lr))

        target_skill_actor = Model.create(actor_def(skill_dim),
                                          inputs=[skill_actor_key, observations])

        # actor takes as input the observation and skill and returns the action
        dummy_skill = jnp.zeros(observations.shape[:-1] + (skill_dim,))

        actor_inp = jnp.concatenate([observations, dummy_skill], axis=-1)

        actor = Model.create(actor_def(action_dim),
                             inputs=[actor_key, actor_inp],
                             tx=optax.adam(learning_rate=actor_lr))

        # initialize a critic for the skill policy and the action critic
        skill_critic_key, critic_key = jax.random.split(critic_key, 2)
        critic_def = critic_net.DoubleCritic(hidden_dims)

        # skill critic takes as input the observation and skill
        skill_critic = Model.create(critic_def,
                                    inputs=[skill_critic_key, observations, dummy_skill],
                                    tx=optax.adam(learning_rate=critic_lr))
        target_skill_critic = Model.create(
            critic_def, inputs=[skill_critic_key, observations, dummy_skill])

        # action critic takes as input the observation, skill and action
        critic = Model.create(critic_def,
                              inputs=[critic_key, actor_inp, actions],
                              tx=optax.adam(learning_rate=critic_lr))
        target_critic = Model.create(
            critic_def, inputs=[critic_key, actor_inp, actions])

        # initialize a temperature for the skill and true policy
        skill_temp_key, temp_key = jax.random.split(temp_key, 2)

        skill_temp = Model.create(temperature.Temperature(init_temperature),
                                  inputs=[skill_temp_key],
                                  tx=optax.adam(learning_rate=temp_lr))

        temp = Model.create(temperature.Temperature(init_temperature),
                            inputs=[temp_key],
                            tx=optax.adam(learning_rate=temp_lr))

        # initialize an ensemble model to predict the next state given the current state and skill
        dyn_ent_temp_key, rng = jax.random.split(rng, 2)
        dyn_ent_temp = Model.create(temperature.Temperature(init_temperature_dyn_entropy),
                                    inputs=[dyn_ent_temp_key],
                                    tx=optax.adamw(learning_rate=dyn_ent_lr, weight_decay=dyn_wd))

        tx = optax.adamw(learning_rate=ens_lr, weight_decay=ens_wd)
        model_key, rng = jax.random.split(rng, 2)

        output_dim = observations.shape[-1]
        if predict_reward:
            output_dim += 1

        if learn_std:
            model_type = ProbabilisticEnsemble
        else:
            model_type = DeterministicEnsemble
        ensemble = model_type(
            model_kwargs={'hidden_dims': model_hidden_dims + (output_dim,),
                          'init_scale': jnp.sqrt(2)},
            optimizer=tx,
            num_heads=self.num_heads)

        ens_state = ensemble.init(key=model_key, input=actor_inp)

        self.actor = actor
        self.skill_actor = skill_actor
        self.target_skill_actor = target_skill_actor

        self.critic = critic
        self.target_critic = target_critic
        self.skill_critic = skill_critic
        self.target_skill_critic = target_skill_critic

        self.temp = temp
        self.skill_temp = skill_temp
        self.dyn_ent_temp = dyn_ent_temp

        self.ensemble = ensemble
        self.ens_state = ens_state

        self.rng = rng

        self.step = 1

    def sample_actions(self,
                       observations: np.ndarray,
                       skill: np.ndarray,
                       temperature: float = 1.0) -> np.ndarray:
        assert skill.shape[-1] == self.skill_dim
        observations = np.concatenate([observations, skill], axis=-1)
        rng, actions = policies.sample_actions(self.rng, self.actor.apply_fn,
                                               self.actor.params, observations,
                                               temperature)
        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def sample_skill(self,
                     observations: np.ndarray,
                     temperature: float = 1.0) -> np.ndarray:

        rng, skill = policies.sample_actions(self.rng, self.skill_actor.apply_fn,
                                             self.skill_actor.params, observations,
                                             temperature)
        self.rng = rng

        skill = np.asarray(skill)
        return np.clip(skill, -1, 1)

    def update(self, batch: Union[NstepSkillBatch, SkillBatch]) -> InfoDict:
        self.step += 1

        new_rng, new_actor, new_skill_actor, new_target_skill_actor, new_critic, new_target_critic, new_skill_critic, \
            new_target_skill_critic, new_temp, new_skill_temp, new_dyn_entropy_temp, new_ens_state, info \
            = _update_jit(
                    rng=self.rng,
                    actor=self.actor,
                    skill_actor=self.skill_actor,
                    target_skill_actor=self.target_skill_actor,
                    critic=self.critic,
                    skill_critic=self.skill_critic,
                    target_critic=self.target_critic,
                    target_skill_critic=self.target_skill_critic,
                    temp=self.temp,
                    skill_temp=self.skill_temp,
                    dyn_entropy_temp=self.dyn_ent_temp,
                    ens_state=self.ens_state,
                    ens=self.ensemble,
                    batch=batch,
                    discount=self.discount,
                    tau=self.tau,
                    target_entropy=self.target_entropy,
                    target_skill_entropy=self.target_skill_entropy,
                    backup_entropy=self.backup_entropy,
                    update_target=self.step % self.target_update_period == 0,
                    use_log_transform=self.use_log_transform,
                    predict_rewards=self.predict_rewards,
                    predict_diff=self.predict_diff,
                    num_skill_samples=self.num_skill_samples,
            )

        self.rng = new_rng

        self.actor = new_actor
        self.skill_actor = new_skill_actor
        self.target_skill_actor = new_target_skill_actor

        self.critic = new_critic
        self.target_critic = new_target_critic
        self.skill_critic = new_skill_critic
        self.target_skill_critic = new_target_skill_critic

        self.temp = new_temp
        self.skill_temp = new_skill_temp
        self.dyn_ent_temp = new_dyn_entropy_temp

        self.ens_state = new_ens_state

        return info
