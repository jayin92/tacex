import functools
from typing import Sequence, Tuple, Optional, Dict

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrl.agents.drq.augmentations import batched_random_crop
from jaxrl2.agents.tactile_agents.networks import TactileMaxInfoDrQv2DoubleCritic
from jaxrl2.agents.tactile_agents.networks import TactileDrQv2Policy
from jaxrl2.agents.maxinfodrqv2.actor import update as update_actor
from jaxrl.agents.drq.drq_learner import target_update
from jaxrl2.agents.tactile_agents.tactile_maxinfodrqv2.critic import update as update_critic
from jaxrl2.data.tactile.buffers import NstepBatch
from jaxrl.networks import policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey
from jaxrl2.utils.tactile.utils import convert_obs_to_tact_obs
from jaxrl.agents.sac.temperature import update as update_temp
from jaxrl.agents.sac import temperature
from jaxrl2.networks.ensemble_model import EnsembleState, DeterministicEnsemble, ProbabilisticEnsemble


def update_batch(batch: NstepBatch, key: PRNGKey):
    def crop_obs(key, obs):
        new_img = batched_random_crop(key, obs.image)
        return obs.replace(image=new_img)
    obs_key, next_obs_key, n_step_next_obs_key = jax.random.split(key, 3)
    observations = crop_obs(obs_key, batch.observations)
    next_observations = crop_obs(next_obs_key, batch.next_observations)
    n_step_next_observations = crop_obs(n_step_next_obs_key, batch.n_step_next_observations)
    batch = batch._replace(observations=observations,
                           next_observations=next_observations,
                           n_step_next_observations=n_step_next_observations,
                           )
    return batch


@functools.partial(jax.jit, static_argnames=('ens',
                                             'update_target',
                                             'use_log_transform',
                                             'predict_rewards',
                                             'predict_diff',
                                             'mask_expl_critic'))
def _update_jit(
        rng: PRNGKey, actor: Model, target_actor: Model,
        critic: Model, target_critic: Model,
        dyn_ent_temp: Model, ens: DeterministicEnsemble, ens_state: EnsembleState,
        batch: NstepBatch, noise_std: float, noise_clip: float,
        discount: float, tau: float,
        update_target: bool, use_log_transform: bool,
        predict_rewards: bool, predict_diff: bool, mask_expl_critic: bool,
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, EnsembleState, InfoDict]:
    rng, key = jax.random.split(rng, 2)
    batch = update_batch(batch, key)

    rng, key = jax.random.split(rng)
    new_critic, ens_state, critic_info = update_critic(key=key,
                                                       actor=actor,
                                                       critic=critic,
                                                       target_critic=target_critic,
                                                       ens=ens,
                                                       ens_state=ens_state,
                                                       batch=batch,
                                                       noise_std=noise_std,
                                                       noise_clip=noise_clip,
                                                       discount=discount,
                                                       mask_expl_critic=mask_expl_critic,
                                                       )

    next_state, state = critic_info['next_state'], critic_info['state']
    image_obs, tact_obs = critic_info['image_obs'], critic_info['tact_obs']
    assert isinstance(state, jnp.ndarray)
    assert isinstance(next_state, jnp.ndarray)
    assert isinstance(image_obs, jnp.ndarray)
    if tact_obs is None:
        use_tact_obs = False
    else:
        use_tact_obs = True
        assert isinstance(tact_obs, jnp.ndarray)
    del critic_info['next_state'], critic_info['state'], critic_info['image_obs'], critic_info['tact_obs']
    if update_target:
        new_target_critic = target_update(new_critic, target_critic, tau)
    else:
        new_target_critic = target_critic

    # Use critic conv layers in actor:
    new_actor_params = actor.params.copy()
    new_actor_params.update({'SharedEncoder': new_critic.params['SharedEncoder']})
    actor = actor.replace(params=new_actor_params)

    rng, key = jax.random.split(rng)
    new_actor, ens_state, actor_info = update_actor(key=key,
                                                    actor=actor,
                                                    target_actor=target_actor,
                                                    critic=new_critic,
                                                    dyn_ent_temp=dyn_ent_temp,
                                                    ens=ens,
                                                    ens_state=ens_state,
                                                    state=state,
                                                    batch=batch,
                                                    noise_std=noise_std,
                                                    noise_clip=noise_clip)

    if update_target:
        new_target_actor = target_update(new_actor, target_actor, tau)
    else:
        new_target_actor = target_actor

    new_dyn_ent_temp, dyn_ent_info = update_temp(dyn_ent_temp, actor_info['info_gain'],
                                                 actor_info['target_info_gain'],
                                                 use_log_transform=use_log_transform)
    dyn_ent_info = {f'dyn_ent_{key}': val for key, val in dyn_ent_info.items()}

    if predict_diff:
        outputs = next_state - state
    else:
        outputs = next_state
    if use_tact_obs:
        outputs = jnp.concatenate([outputs, image_obs, tact_obs], axis=-1)
    else:
        outputs = jnp.concatenate([outputs, image_obs], axis=-1)
    if predict_rewards:
        outputs = jnp.concatenate([outputs, batch.rewards.reshape(-1, 1)], axis=-1)
    new_ens_state, (loss, mse) = ens.update(
        input=jnp.concatenate([state, batch.actions], axis=-1),
        output=outputs,
        state=ens_state,
    )
    ens_info = {'ens_nll': loss,
                'ens_mse': mse,
                'ens_inp_mean': ens_state.ensemble_normalizer_state.input_normalizer_state.mean.mean(),
                'ens_inp_std': ens_state.ensemble_normalizer_state.input_normalizer_state.std.mean(),
                'ens_out_mean': ens_state.ensemble_normalizer_state.output_normalizer_state.mean.mean(),
                'ens_out_std': ens_state.ensemble_normalizer_state.output_normalizer_state.std.mean(),
                'ens_info_gain_mean': ens_state.ensemble_normalizer_state.info_gain_normalizer_state.mean.mean(),
                'ens_info_gain_std': ens_state.ensemble_normalizer_state.info_gain_normalizer_state.std.mean(),
                }

    actor_info['noise_std'] = noise_std
    actor_info['noise_clip'] = noise_clip

    return rng, \
        new_actor, \
        new_critic, \
        new_target_actor, \
        new_target_critic, \
        new_dyn_ent_temp, \
        new_ens_state, {
        **critic_info,
        **actor_info,
        **dyn_ent_info,
        **ens_info,
    }


class TactileMaxInfoDrQv2Learner(object):

    def __init__(self,
                 seed: int,
                 observations: Dict,
                 actions: jnp.ndarray,
                 actor_lr: float = 1e-4,
                 critic_lr: float = 1e-4,
                 dyn_ent_lr: float = 1e-4,
                 dyn_wd: float = 0.0,
                 ens_lr: float = 1e-4,
                 ens_wd: float = 0.0,
                 hidden_dims: Sequence[int] = (256, 256),
                 model_hidden_dims: Sequence[int] = (256, 256),
                 cnn_features: Sequence[int] = (32, 32, 32, 32),
                 cnn_strides: Sequence[int] = (2, 1, 1, 1),
                 cnn_padding: str = 'VALID',
                 latent_dim: int = 50,
                 obs_dim: int = 32,
                 num_heads: int = 5,
                 predict_reward: bool = True,
                 predict_diff: bool = False,
                 use_log_transform: bool = True,
                 learn_std: bool = False,
                 mask_expl_critic: bool = False,
                 discount: float = 0.99,
                 tau: float = 0.005,
                 target_update_period: int = 1,
                 init_sig: float = 0.2,
                 final_sig: float = 0.2,
                 steps_to_final_sig: int = 500_000,
                 noise_clip: float = 0.3,
                 init_temperature_dyn_entropy: float = 1.0,
                 max_gradient_norm: Optional[float] = None,
                 max_ensemble_gradient_norm: Optional[float] = None,
                 mask_touch: bool = False,
                 ):

        observations = convert_obs_to_tact_obs(observations)
        action_dim = actions.shape[-1]

        self.predict_reward = predict_reward
        self.predict_diff = predict_diff
        self.num_heads = num_heads
        self.mask_expl_critic = mask_expl_critic

        self.tau = tau
        self.target_update_period = target_update_period
        self.discount = discount

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key = jax.random.split(rng, 3)

        actor_def = TactileDrQv2Policy(hidden_dims, action_dim, cnn_features,
                                       cnn_strides, cnn_padding, latent_dim, mask_touch=mask_touch)
        actor_optimizer = optax.adam(learning_rate=actor_lr)
        critic_optimizer = optax.adam(learning_rate=critic_lr)
        dyn_ent_temp_optimizer = optax.adamw(learning_rate=dyn_ent_lr, weight_decay=dyn_wd)
        if max_gradient_norm:
            assert max_gradient_norm > 0
            actor_optimizer = optax.chain(
                optax.clip_by_global_norm(max_gradient_norm),  # Apply gradient clipping
                actor_optimizer  # Apply Adam optimizer
            )
            critic_optimizer = optax.chain(
                optax.clip_by_global_norm(max_gradient_norm),
                critic_optimizer,
            )

            dyn_ent_temp_optimizer = optax.chain(
                optax.clip_by_global_norm(max_gradient_norm),
                dyn_ent_temp_optimizer,
            )
        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=actor_optimizer)

        target_actor = Model.create(
            actor_def, inputs=[actor_key, observations])

        critic_def = TactileMaxInfoDrQv2DoubleCritic(hidden_dims, cnn_features, cnn_strides,
                                                     cnn_padding, latent_dim, obs_dim=obs_dim,
                                                     mask_touch=mask_touch)
        critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              tx=critic_optimizer)
        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions])

        dyn_ent_temp_key, rng = jax.random.split(rng, 2)
        dyn_ent_temp = Model.create(temperature.Temperature(init_temperature_dyn_entropy),
                                    inputs=[dyn_ent_temp_key],
                                    tx=dyn_ent_temp_optimizer)

        model_optimizer = optax.adamw(learning_rate=ens_lr, weight_decay=ens_wd)
        if max_ensemble_gradient_norm:
            assert max_ensemble_gradient_norm > 0
            model_optimizer = optax.chain(
                optax.clip_by_global_norm(max_ensemble_gradient_norm),
                model_optimizer,
            )
        model_key, rng = jax.random.split(rng, 2)

        output_dim = latent_dim + obs_dim * (1 + int(observations.tactile is not None))

        if predict_reward:
            output_dim += 1

        if learn_std:
            model_type = ProbabilisticEnsemble
        else:
            model_type = DeterministicEnsemble
        ensemble = model_type(
            model_kwargs={'hidden_dims': model_hidden_dims + (output_dim,),
                          'init_scale': jnp.sqrt(2)},
            optimizer=model_optimizer,
            num_heads=self.num_heads)

        dummy_state = jnp.zeros((actions.shape[0], latent_dim))
        ens_state = ensemble.init(key=model_key, input=jnp.concatenate([dummy_state, actions], axis=-1))

        self.use_log_transform = use_log_transform

        self.actor = actor
        self.target_actor = target_actor
        self.critic = critic
        self.target_critic = target_critic
        self.dyn_ent_temp = dyn_ent_temp
        self.ens = ensemble
        self.ens_state = ens_state

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
        new_rng, new_actor, new_critic, new_target_actor, new_target_critic, new_dyn_ent_temp, new_ens_state, info \
            = _update_jit(
            rng=self.rng,
            actor=self.actor,
            target_actor=self.target_actor,
            critic=self.critic,
            target_critic=self.target_critic,
            dyn_ent_temp=self.dyn_ent_temp,
            ens=self.ens,
            ens_state=self.ens_state,
            batch=batch,
            noise_std=noise_std,
            noise_clip=self.noise_clip,
            discount=self.discount,
            tau=self.tau,
            update_target=self.step % self.target_update_period == 0,
            use_log_transform=self.use_log_transform,
            predict_rewards=self.predict_reward,
            predict_diff=self.predict_diff,
            mask_expl_critic=self.mask_expl_critic)

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.target_actor = new_target_actor
        self.target_critic = new_target_critic
        self.dyn_ent_temp = new_dyn_ent_temp
        self.ens_state = new_ens_state

        return info
