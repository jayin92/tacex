"""Implementations of algorithms for continuous control."""

import functools
from typing import Sequence, Tuple, Dict

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrl.agents.drq.augmentations import batched_random_crop
from jaxrl2.agents.tactile_agents.networks import TactileDrQDoubleCritic
from jaxrl2.agents.tactile_agents.networks import TactileDrQv2Policy
from jaxrl2.utils.tactile.utils import TactileObs, convert_obs_to_tact_obs
from jaxrl2.agents.tactile_agents.tactile_drqv2.actor import update as update_actor
from jaxrl.agents.drq.drq_learner import target_update
from jaxrl2.agents.tactile_agents.tactile_drqv2.critic import update as update_critic
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
    aug_image = batched_random_crop(key, batch.observations.image)
    rng, key = jax.random.split(rng)
    n_step_next_aug_image = batched_random_crop(key, batch.n_step_next_observations.image)

    observations = batch.observations.replace(
        image=aug_image,
    )

    n_step_next_observations = batch.n_step_next_observations.replace(
        image=n_step_next_aug_image,
    )

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


class TactileDrQv2Learner(object):

    def __init__(self,
                 seed: int,
                 observations: Dict,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 cnn_features: Sequence[int] = (32, 32, 32, 32),
                 cnn_strides: Sequence[int] = (2, 1, 1, 1),
                 cnn_padding: str = 'VALID',
                 latent_dim: int = 50,
                 discount: float = 0.99,
                 tau: float = 0.005,
                 target_update_period: int = 1,
                 init_sig: float = 0.2,
                 final_sig: float = 0.2,
                 steps_to_final_sig: int = 500_000,
                 noise_clip: float = 0.3,
                 mask_touch: bool = False,
                 ):
        observations = convert_obs_to_tact_obs(observations)
        action_dim = actions.shape[-1]

        self.tau = tau
        self.target_update_period = target_update_period
        self.discount = discount

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key = jax.random.split(rng, 3)

        # use tactile policy with the tactile encoder
        actor_def = TactileDrQv2Policy(hidden_dims, action_dim, cnn_features,
                                       cnn_strides, cnn_padding, latent_dim, mask_touch=mask_touch)
        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=optax.adam(learning_rate=actor_lr))

        # use tactile critic with the tactile encoder
        critic_def = TactileDrQDoubleCritic(hidden_dims, cnn_features, cnn_strides,
                                            cnn_padding, latent_dim, mask_touch=mask_touch)
        critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              tx=optax.adam(learning_rate=critic_lr))
        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions])

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
                       observations: Dict,
                       temperature: float = 1.0) -> np.ndarray:
        tact_obs = convert_obs_to_tact_obs(observations)
        rng, actions = policies.sample_actions(self.rng,
                                               self.actor.apply_fn,
                                               self.actor.params,
                                               tact_obs,
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
        return noise_scale

    def update(self, batch: NstepBatch) -> InfoDict:
        noise_std = self.noise_scale
        self.step += 1
        obs, next_obs = convert_obs_to_tact_obs(batch.observations, batch.next_observations)
        next_n_step_obs = convert_obs_to_tact_obs(batch.n_step_next_observations)
        batch = batch._replace(observations=obs, next_observations=next_obs, n_step_next_observations=next_n_step_obs)
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

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.target_critic = new_target_critic

        return info
