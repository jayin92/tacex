"""Implementations of algorithms for continuous control."""

import functools
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrl.agents.drq.augmentations import batched_random_crop
from jaxrl.agents.drq.networks import DrQPolicy
from jaxrl2.agents.maxinfodrq.networks import MaxInfoDrQDoubleCritic
from jaxrl.agents.sac import temperature
from jaxrl2.agents.maxinfodrq.actor import update as update_actor
from jaxrl.agents.drq.drq_learner import target_update
from jaxrl2.agents.maxinfodrq.critic import update as update_critic
from jaxrl.datasets import Batch
from jaxrl.networks import policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey
from jaxrl.agents.sac.temperature import update as update_temp
from jaxrl2.networks.ensemble_model import EnsembleState, DeterministicEnsemble, ProbabilisticEnsemble


@functools.partial(jax.jit,
                   static_argnames=('ens',
                                    'update_target',
                                    'use_log_transform',
                                    'predict_rewards',
                                    'predict_diff',))
def _update_jit(
        rng: PRNGKey, actor: Model, critic: Model, target_actor: Model, target_critic: Model, temp: Model,
        dyn_entropy_temp: Model, ens: DeterministicEnsemble, ens_state: EnsembleState,
        batch: Batch, discount: float, tau: float,
        target_entropy: float, update_target: bool,
        use_log_transform: bool, predict_rewards: bool, predict_diff: bool,
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, Model, EnsembleState, InfoDict]:
    rng, key = jax.random.split(rng)
    observations = batched_random_crop(key, batch.observations)
    rng, key = jax.random.split(rng)
    next_observations = batched_random_crop(key, batch.next_observations)

    batch = batch._replace(observations=observations,
                           next_observations=next_observations)

    rng, key = jax.random.split(rng)
    new_critic, ens_state, critic_info = update_critic(
        key=key,
        actor=actor,
        critic=critic,
        target_critic=target_critic,
        temp=temp,
        dyn_entropy_temp=dyn_entropy_temp,
        ens=ens,
        ens_state=ens_state,
        batch=batch,
        discount=discount,
        backup_entropy=True)

    next_state, state, obs = critic_info['next_state'], critic_info['state'], critic_info['obs']
    assert isinstance(state, jnp.ndarray)
    assert isinstance(next_state, jnp.ndarray)
    assert isinstance(obs, jnp.ndarray)
    del critic_info['next_state'], critic_info['state'], critic_info['obs']
    if update_target:
        new_target_critic = target_update(new_critic, target_critic, tau)
    else:
        new_target_critic = target_critic

    # update encoder params for the actors
    new_actor_params = actor.params.copy()
    new_actor_params.update({'SharedEncoder': new_critic.params['SharedEncoder']})
    actor = actor.replace(params=new_actor_params)

    # update encoder params for the target actor
    # new_actor_params = target_actor.params.copy()
    # new_actor_params.update({'SharedEncoder': new_target_critic.params['SharedEncoder']})
    # target_actor = target_actor.replace(params=new_actor_params)

    rng, key = jax.random.split(rng)
    new_actor, ens_state, actor_info = update_actor(key=key,
                                                    actor=actor,
                                                    target_actor=target_actor,
                                                    critic=new_critic,
                                                    temp=temp,
                                                    dyn_entropy_temp=dyn_entropy_temp,
                                                    ens=ens,
                                                    ens_state=ens_state,
                                                    state=state,
                                                    batch=batch)

    if update_target:
        new_target_actor = target_update(new_actor, target_actor, tau)
    else:
        new_target_actor = target_actor

    new_temp, alpha_info = update_temp(temp, actor_info['entropy'],
                                       target_entropy, use_log_transform=use_log_transform)
    new_dyn_entropy_temp, dyn_ent_info = update_temp(dyn_entropy_temp, actor_info['info_gain'],
                                                     actor_info['target_info_gain'],
                                                     use_log_transform=use_log_transform)
    dyn_ent_info = {f'dyn_ent_{key}': val for key, val in dyn_ent_info.items()}

    if predict_diff:
        outputs = next_state - state
    else:
        outputs = next_state
    outputs = jnp.concatenate([outputs, obs], axis=-1)
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

    return rng, \
        new_actor, \
        new_critic, \
        new_target_actor, \
        new_target_critic, \
        new_temp, \
        new_dyn_entropy_temp, \
        new_ens_state, {
        **critic_info,
        **actor_info,
        **alpha_info,
        **dyn_ent_info,
        **ens_info,
    }


class MaxInfoDrQLearner(object):

    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 dyn_ent_lr: float = 3e-4,
                 dyn_wd: float = 0.0,
                 ens_lr: float = 3e-4,
                 ens_wd: float = 0.0,
                 hidden_dims: Sequence[int] = (256, 256),
                 model_hidden_dims: Sequence[int] = (256, 256),
                 cnn_features: Sequence[int] = (32, 32, 32, 32),
                 cnn_strides: Sequence[int] = (2, 1, 1, 1),
                 cnn_padding: str = 'VALID',
                 latent_dim: int = 50,
                 obs_dim: int = 64,
                 num_heads: int = 5,
                 predict_reward: bool = True,
                 predict_diff: bool = False,
                 use_log_transform: bool = True,
                 learn_std: bool = False,
                 discount: float = 0.99,
                 tau: float = 0.005,
                 target_update_period: int = 1,
                 target_entropy: Optional[float] = None,
                 init_temperature_dyn_entropy: float = 1.0,
                 init_temperature: float = 0.1):

        self.predict_reward = predict_reward
        self.predict_diff = predict_diff
        self.num_heads = num_heads

        action_dim = actions.shape[-1]

        if target_entropy is None:
            self.target_entropy = -action_dim
        else:
            self.target_entropy = target_entropy

        self.tau = tau
        self.target_update_period = target_update_period
        self.discount = discount

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        actor_def = DrQPolicy(hidden_dims, action_dim, cnn_features,
                              cnn_strides, cnn_padding, latent_dim)
        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=optax.adam(learning_rate=actor_lr))

        target_actor = Model.create(actor_def,
                                    inputs=[actor_key, observations])

        critic_def = MaxInfoDrQDoubleCritic(hidden_dims, cnn_features, cnn_strides,
                                           cnn_padding, latent_dim, obs_dim=obs_dim)
        critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              tx=optax.adam(learning_rate=critic_lr))
        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions])

        temp = Model.create(temperature.Temperature(init_temperature),
                            inputs=[temp_key],
                            tx=optax.adam(learning_rate=temp_lr))
        # dummy_out = critic(observations, actions)

        # dummy_state, dummy_obs = dummy_out[-2], dummy_out[-1]
        # information gain kwargs
        dyn_ent_temp_key, rng = jax.random.split(rng, 2)
        dyn_ent_temp = Model.create(temperature.Temperature(init_temperature_dyn_entropy),
                                    inputs=[dyn_ent_temp_key],
                                    tx=optax.adamw(learning_rate=dyn_ent_lr, weight_decay=dyn_wd))

        tx = optax.adamw(learning_rate=ens_lr, weight_decay=ens_wd)
        model_key, rng = jax.random.split(rng, 2)

        output_dim = latent_dim + obs_dim

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

        dummy_state = jnp.zeros((actions.shape[0], latent_dim))
        ens_state = ensemble.init(key=model_key, input=jnp.concatenate([dummy_state, actions], axis=-1))

        self.use_log_transform = use_log_transform

        self.actor = actor
        self.target_actor = target_actor
        self.critic = critic
        self.target_critic = target_critic
        self.temp = temp
        self.dyn_ent_temp = dyn_ent_temp
        self.ens_state = ens_state
        self.ensemble = ensemble
        self.rng = rng

        self.step = 1

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> np.ndarray:
        rng, actions = policies.sample_actions(self.rng, self.actor.apply_fn,
                                               self.actor.params, observations,
                                               temperature)

        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch) -> InfoDict:
        self.step += 1
        new_rng, new_actor, new_critic, new_target_actor, new_target_critic, \
            new_temp, new_dyn_entropy_temp, new_ens_state, info = _update_jit(
                rng=self.rng,
                actor=self.actor,
                critic=self.critic,
                target_actor=self.target_actor,
                target_critic=self.target_critic,
                temp=self.temp,
                dyn_entropy_temp=self.dyn_ent_temp,
                ens=self.ensemble,
                ens_state=self.ens_state,
                batch=batch,
                discount=self.discount,
                tau=self.tau,
                target_entropy=self.target_entropy,
                update_target=self.step % self.target_update_period == 0,
                use_log_transform=self.use_log_transform,
                predict_rewards=self.predict_reward,
                predict_diff=self.predict_diff)

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.target_actor = new_target_actor
        self.target_critic = new_target_critic
        self.temp = new_temp
        self.dyn_ent_temp = new_dyn_entropy_temp
        self.ens_state = new_ens_state

        return info
