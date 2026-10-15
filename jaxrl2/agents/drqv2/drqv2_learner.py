"""Implementations of algorithms for continuous control."""

import functools
from typing import Sequence, Tuple, Callable, Dict, Optional
import copy
import chex
from jaxtyping import PyTree

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrl.agents.drq.augmentations import batched_random_crop
from jaxrl.agents.drq.networks import DrQDoubleCritic
from jaxrl2.agents.drqv2.networks import DrQv2Policy, DrQv2PolicyDormantRatioCalculator, perturb_params
from jaxrl2.agents.drqv2.actor import update as update_actor
from jaxrl.agents.drq.drq_learner import target_update
from jaxrl2.agents.drqv2.critic import update as update_critic
from jaxrl2.data import NstepBatch
from jaxrl.networks import policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey


@functools.partial(jax.jit, static_argnames='update_target')
def _update_jit(
        rng: PRNGKey, actor: Model, critic: Model, target_critic: Model,
        batch: NstepBatch, noise_std: float, noise_clip: float,
        discount: float, tau: float,
        update_target: bool,
) -> Tuple[PRNGKey, Model, Model, Model, InfoDict]:
    rng, key = jax.random.split(rng)
    observations = batched_random_crop(key, batch.observations)
    rng, key = jax.random.split(rng)
    n_step_next_observations = batched_random_crop(key, batch.n_step_next_observations)

    batch = batch._replace(observations=observations,
                           n_step_next_observations=n_step_next_observations)

    rng, key = jax.random.split(rng)
    new_critic, critic_info = update_critic(key=key,
                                            actor=actor,
                                            critic=critic,
                                            target_critic=target_critic,
                                            batch=batch,
                                            noise_std=noise_std,
                                            noise_clip=noise_clip,
                                            discount=discount,
                                            )
    if update_target:
        new_target_critic = target_update(new_critic, target_critic, tau)
    else:
        new_target_critic = target_critic

    # Use critic conv layers in actor:
    new_actor_params = actor.params.copy()
    new_actor_params.update({'SharedEncoder': new_critic.params['SharedEncoder']})
    actor = actor.replace(params=new_actor_params)

    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor(key=key,
                                         actor=actor,
                                         critic=new_critic,
                                         batch=batch,
                                         noise_std=noise_std,
                                         noise_clip=noise_clip)
    actor_info['noise_std'] = noise_std
    actor_info['noise_clip'] = noise_clip
    return rng, new_actor, new_critic, new_target_critic, {
        **critic_info,
        **actor_info,
    }


class DormantRatioModel(object):
    def __init__(self,
                 dormant_ratio_calculator: DrQv2PolicyDormantRatioCalculator,
                 observation: chex.Array,
                 action: chex.Array,
                 actor_init_fn: Callable,
                 critic_init_fn: Callable,
                 actor_init_opt_state: PyTree,
                 critic_init_opt_state: PyTree,
                 alpha_min: float = 0.2,
                 alpha_max: float = 0.9,
                 perturb_rate: float = 2.0,
                 target_dormant_ratio: float = 0.2,
                 dormant_temp: float = 10.0,
                 perturbation_freq: int = 2e5,
                 min_awaken_update_steps: int = 2_000,
                 ):
        # self.dormant_ratio_calculator = dormant_ratio_calculator
        self.calc_dormant_ratio = jax.jit(lambda params, obs: dormant_ratio_calculator.apply(
            {'params': dormant_ratio_calculator.convert_params(params)},
            obs
        ))
        self._actor_init_fn = jax.jit(actor_init_fn)
        self._critic_init_fn = jax.jit(critic_init_fn)
        self._actor_init_opt_state = actor_init_opt_state
        self._critic_init_opt_state = critic_init_opt_state
        self._dormant_ratio = 0.0
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.perturb_rate = perturb_rate
        self.target_dormant_ratio = target_dormant_ratio
        self.dormant_temp = dormant_temp
        self.perturbation_freq = perturbation_freq
        self.awaken_step = -1
        self.min_awaken_update_steps = min_awaken_update_steps
        self._setup_perturbation_dictionary(observation=observation, action=action)

    def _setup_perturbation_dictionary(self, observation: chex.Array, action: chex.Array):
        dummy_actor_params = self.get_actor_init_params(rng=jax.random.PRNGKey(0), observation=observation)
        dummy_actor_params = dummy_actor_params.pop('params')
        self._actor_perturbation_weights = jax.tree_util.tree_map(lambda x: 0.0, dummy_actor_params)
        self._actor_perturbation_weights['Dense_0'] = jax.tree_util.tree_map(lambda x: 1.0,
                                                                             dummy_actor_params['Dense_0'])
        self._actor_perturbation_weights['MSEPolicy_0'] = jax.tree_util.tree_map(lambda x: 1.0,
                                                                                 dummy_actor_params[
                                                                                     'MSEPolicy_0'])

        dummy_critic_params = self.get_critic_init_params(rng=jax.random.PRNGKey(0),
                                                          observation=observation, action=action)
        dummy_critic_params = dummy_critic_params.pop('params')
        self._critic_perturbation_weights = jax.tree_util.tree_map(lambda x: 0.0, dummy_critic_params)
        self._critic_perturbation_weights['Dense_0'] = jax.tree_util.tree_map(lambda x: 1.0,
                                                                              dummy_critic_params['Dense_0'])
        self._critic_perturbation_weights['DoubleCritic_0'] = jax.tree_util.tree_map(lambda x: 1.0,
                                                                                     dummy_critic_params[
                                                                                         'DoubleCritic_0'])

    def get_actor_init_params(self, rng: chex.Array, observation: chex.Array):
        return self._actor_init_fn(rng, observation)

    def get_critic_init_params(self, rng: chex.Array, observation: chex.Array, action: chex.Array):
        return self._critic_init_fn(rng, observation, action)

    def update_dormant_ratio(self, actor: Model, observation: chex.Array):
        self._dormant_ratio = self.calc_dormant_ratio(actor.params, observation)

    def perturb(self, actor: Model, critic: Model, target_critic: Model,
                observation: chex.Array, action: chex.Array, rng: chex.Array,
                step: int,
                ):
        if step >= 1 and step % self.perturbation_freq == 0:
            actor_rng, critic_rng, target_critic_rng = jax.random.split(rng, 3)
            init_actor_params = self.get_actor_init_params(rng=actor_rng, observation=observation)
            init_actor_params = init_actor_params.pop('params')
            init_critic_params = self.get_critic_init_params(rng=critic_rng, observation=observation, action=action)
            init_critic_params = init_critic_params.pop('params')
            init_target_critic_params = self.get_critic_init_params(rng=target_critic_rng,
                                                                    observation=observation, action=action)
            init_target_critic_params = init_target_critic_params.pop('params')

            perturb_factor = self.perturb_factor

            new_actor_params = perturb_params(
                init_params=init_actor_params,
                trained_params=actor.params,
                perturb_factor=perturb_factor,
                perturb_keys=self.actor_perturbation_params,
            )

            new_critic_params = perturb_params(
                init_params=init_critic_params,
                trained_params=critic.params,
                perturb_factor=perturb_factor,
                perturb_keys=self.critic_perturbation_params,
            )

            new_target_critic_params = perturb_params(
                init_params=init_target_critic_params,
                trained_params=target_critic.params,
                perturb_factor=perturb_factor,
                perturb_keys=self.critic_perturbation_params,
            )

            new_actor = actor.replace(
                params=new_actor_params,
                opt_state=self.actor_init_opt_state
            )

            new_critic = critic.replace(
                params=new_critic_params,
                opt_state=self.critic_init_opt_state
            )
            new_target_critic = target_critic.replace(params=new_target_critic_params)
            return new_actor, new_critic, new_target_critic
        else:
            return actor, critic, target_critic

    def update_awaken_step(self, step: int):
        if self.awaken_step == -1 and self.dormant_ratio < self.target_dormant_ratio \
                and self.min_awaken_update_steps < step:
            self.awaken_step = step

    @property
    def actor_perturbation_params(self):
        return self._actor_perturbation_weights

    @property
    def critic_perturbation_params(self):
        return self._critic_perturbation_weights

    @property
    def actor_init_opt_state(self):
        return self._actor_init_opt_state

    @property
    def critic_init_opt_state(self):
        return self._critic_init_opt_state

    @property
    def dormant_ratio(self):
        return self._dormant_ratio

    @property
    def dormant_noise_scale(self):
        dorm_frac = (self.dormant_ratio - self.target_dormant_ratio) * self.dormant_temp
        noise_scale_awakened = 1.0 / (1.0 + jnp.exp(-dorm_frac))
        return noise_scale_awakened

    @property
    def perturb_factor(self):
        perturb_factor = jnp.clip(1 - self.perturb_rate * self.dormant_ratio, min=self.alpha_min,
                                  max=self.alpha_max)
        return perturb_factor


class DrQv2Learner(object):

    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 1e-4,
                 critic_lr: float = 1e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 cnn_features: Sequence[int] = (32, 32, 32, 32),
                 cnn_strides: Sequence[int] = (2, 1, 1, 1),
                 cnn_padding: str = 'VALID',
                 latent_dim: int = 50,
                 discount: float = 0.99,
                 tau: float = 0.01,
                 target_update_period: int = 1,
                 init_sig: float = 0.2,
                 final_sig: float = 0.2,
                 steps_to_final_sig: int = 500_000,
                 noise_clip: float = 0.3,
                 use_dormant_ratio: bool = False,
                 dormant_ratio_model_kwargs: Optional[Dict] = None,
                 ):

        action_dim = actions.shape[-1]

        self.tau = tau
        self.target_update_period = target_update_period
        self.discount = discount

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key = jax.random.split(rng, 3)

        actor_def = DrQv2Policy(hidden_dims, action_dim, cnn_features,
                                cnn_strides, cnn_padding, latent_dim)
        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=optax.adam(learning_rate=actor_lr))

        critic_def = DrQDoubleCritic(hidden_dims, cnn_features, cnn_strides,
                                     cnn_padding, latent_dim)
        critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              tx=optax.adam(learning_rate=critic_lr))
        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions])

        if use_dormant_ratio:
            dormant_ratio_calculator = DrQv2PolicyDormantRatioCalculator(
                hidden_dims, action_dim, cnn_features,
                cnn_strides, cnn_padding, latent_dim,
            )
            if dormant_ratio_model_kwargs is None:
                dormant_ratio_model_kwargs = {}
            self.dormant_ratio_model = DormantRatioModel(
                dormant_ratio_calculator=dormant_ratio_calculator,
                observation=observations,
                action=actions,
                actor_init_fn=actor_def.init,
                critic_init_fn=critic_def.init,
                actor_init_opt_state=copy.deepcopy(actor.opt_state),
                critic_init_opt_state=copy.deepcopy(critic.opt_state),
                **dormant_ratio_model_kwargs,
            )
        else:
            self.dormant_ratio_model = None

        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.rng = rng
        self.step = 0

        self.noise_schedule = optax.linear_schedule(
            init_value=init_sig,
            end_value=final_sig,
            transition_steps=steps_to_final_sig,
        )
        self.noise_clip = noise_clip

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> np.ndarray:

        rng, actions = policies.sample_actions(self.rng,
                                               self.actor.apply_fn,
                                               self.actor.params,
                                               observations,
                                               temperature,
                                               distribution='det')
        self.rng = rng
        noise_scale = self.noise_scale
        actions = np.asarray(actions)
        noise = np.random.normal(size=actions.shape) * noise_scale
        actions = actions + noise * temperature
        return np.clip(actions, -1, 1)

    @property
    def noise_scale(self):
        noise_scale = self.noise_schedule(self.step)
        if self.has_dormant_ratio_model:
            if self.dormant_ratio_model.awaken_step >= 0:
                noise_scale = jnp.maximum(self.dormant_ratio_model.dormant_noise_scale, noise_scale)
            else:
                noise_scale = self.dormant_ratio_model.dormant_noise_scale
        return noise_scale

    def update(self, batch: NstepBatch) -> InfoDict:
        noise_std = self.noise_scale
        self.step += 1
        new_rng, new_actor, new_critic, new_target_critic, info = _update_jit(
            rng=self.rng,
            actor=self.actor,
            critic=self.critic,
            target_critic=self.target_critic,
            batch=batch,
            noise_std=noise_std,
            noise_clip=self.noise_clip,
            discount=self.discount,
            tau=self.tau,
            update_target=self.step % self.target_update_period == 0,
        )

        if self.has_dormant_ratio_model:
            self.dormant_ratio_model.update_dormant_ratio(
                actor=new_actor,
                observation=batch.observations,
            )
            new_rng, rng = jax.random.split(new_rng, 2)
            new_actor, new_critic, new_target_critic = self.dormant_ratio_model.perturb(
                actor=new_actor,
                critic=new_critic,
                target_critic=new_target_critic,
                observation=batch.observations,
                action=batch.actions,
                rng=rng,
                step=self.step
            )
            self.dormant_ratio_model.update_awaken_step(step=self.step)
            info['actor_dormant_ratio'] = self.dormant_ratio_model.dormant_ratio
            info['actor_dormant_awaken_step'] = self.dormant_ratio_model.awaken_step
            info['actor_dormant_perturb_factor'] = self.dormant_ratio_model.perturb_factor

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.target_critic = new_target_critic

        return info

    @property
    def has_dormant_ratio_model(self):
        return self.dormant_ratio_model is not None
