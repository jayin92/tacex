"""Implementations of algorithms for continuous control."""

import functools
from typing import Optional, Sequence, Tuple, Dict

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrl.agents.drq.augmentations import batched_random_crop
from jaxrl2.agents.tactile_agents.networks import TactileMaxInfoDrQDoubleCritic, TactileDrQPolicy
from jaxrl2.utils.tactile.utils import convert_obs_to_tact_obs, TactileObs
from jaxrl.agents.sac import temperature
from jaxrl2.agents.drq_explorer.actor import update as update_actor
from jaxrl2.agents.drq_explorer.critic import update as update_critic
from jaxrl.datasets import Batch
from jaxrl.networks import policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey
from jaxrl2.networks.ensemble_model import EnsembleState, DeterministicEnsemble, ProbabilisticEnsemble, \
    NormalizerState, Normalizer


def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_util.tree_map(
        lambda x, y: x * (1 - tau) + y * tau, target_critic.params,
        critic.params)
    # copy over the encoder params for the target --> they all share the same encoder
    new_target_params.update({'SharedEncoder': critic.params['SharedEncoder']})
    return target_critic.replace(params=new_target_params)


def target_update_with_decoder(critic: Model, target_critic: Model, tau: float) -> Model:
    target_critic = target_update(critic, target_critic, tau)
    new_params = target_critic.params
    # copy over the encoder params for the target --> they all share the same encoder
    new_params.update({'SharedDecoder': critic.params['SharedDecoder']})
    return target_critic.replace(params=new_params)


@functools.partial(jax.jit, static_argnames=('ens', 'predict_rewards', 'predict_diff'))
def update_ensemble(batch: Batch,
                    target_critic: Model,
                    ens: DeterministicEnsemble,
                    ens_state: EnsembleState,
                    predict_rewards: bool,
                    predict_diff: bool,
                    ) -> Tuple[EnsembleState, InfoDict]:
    # take dummy actions to get the next state embedding (doesn't depend on the next actions).
    batch_size = batch.actions.shape[0]
    next_actions = batch.actions
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
    next_state = state[:batch_size]
    if has_tactile_obs:
        state, image_obs, tact_obs = state[batch_size:], embed.img_embed[batch_size:], embed.tact_embed[batch_size:]
    else:
        state, image_obs, tact_obs = state[batch_size:], embed.img_embed[batch_size:], None

    # expl_rew, ens_state = ens.get_info_gain(input=jnp.concatenate([state, batch.actions], axis=-1),
    #                                        state=ens_state,
    #                                        update_normalizer=True)

    if predict_diff:
        outputs = next_state - state
    else:
        outputs = next_state
    if has_tactile_obs:
        outputs = jnp.concatenate([outputs, image_obs, tact_obs], axis=-1)
        embedding_info = {
            'img_embed': jnp.mean(image_obs),
            'tact_embed': jnp.mean(tact_obs),
        }
    else:
        outputs = jnp.concatenate([outputs, image_obs], axis=-1)
        embedding_info = {
            'img_embed': jnp.mean(image_obs),
        }
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
                **embedding_info,
                }
    # returns exploration rewards for the exploration critic and policy
    return new_ens_state, ens_info  # , batch._replace(rewards=expl_rew)


@functools.partial(jax.jit, static_argnames='ens')
def get_expl_batch(batch: Batch,
                   ens: DeterministicEnsemble,
                   target_critic: Model,
                   ens_state: EnsembleState) -> Tuple[EnsembleState, Batch]:
    q, state, embed = target_critic(batch.observations, batch.actions)

    expl_rew, new_ens_state = ens.get_info_gain(input=jnp.concatenate([state, batch.actions], axis=-1),
                                                state=ens_state,
                                                update_normalizer=True)
    return new_ens_state, batch._replace(rewards=expl_rew)


def update_critic_with_rc_loss(key: PRNGKey, actor: Model, critic: Model, target_critic: Model,
                               temp: Model, batch: Batch, discount: float,
                               tactile_label: jnp.ndarray,
                               rc_loss_coefficient: float,
                               backup_entropy: bool) -> Tuple[Model, InfoDict]:
    dist = actor(batch.next_observations)
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)
    next_q, _, _ = target_critic(batch.next_observations, next_actions)
    next_q = jnp.min(next_q, axis=0)

    target_q = batch.rewards + discount * batch.masks * next_q

    if backup_entropy:
        ent_coef, _ = temp()
        target_q -= discount * batch.masks * ent_coef * next_log_probs

    def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, InfoDict]:
        q, _, embed = critic.apply_fn({'params': critic_params}, batch.observations,
                                      batch.actions)
        y_pred = embed.full_tact.reshape(tactile_label.shape)
        rc_loss = jnp.mean((tactile_label - y_pred) ** 2)

        q1, q2 = q[0], q[-1]
        critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
        total_loss = critic_loss + rc_loss_coefficient * rc_loss
        return total_loss, {
            'total_loss': total_loss,
            'rc_loss': rc_loss,
            'critic_loss': critic_loss,
            'q1': q1.mean(),
            'q2': q2.mean()
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info


@functools.partial(jax.jit, static_argnames=('update_target', 'use_log_transform', 'tactile_normalizer',
                                             'rc_loss_coefficient'))
def _update_jit(
        rng: PRNGKey, actor: Model, critic: Model, target_critic: Model,
        temp: Model, batch: Batch, discount: float, tau: float,
        target_entropy: float, update_target: bool, use_log_transform: bool,
        rc_loss_coefficient: float, tactile_normalizer_state: NormalizerState, tactile_normalizer: Normalizer,
) -> Tuple[PRNGKey, Model, Model, Model, Model, NormalizerState, InfoDict]:
    # augment images
    rng, key = jax.random.split(rng)
    aug_image = batched_random_crop(key, batch.observations.image)
    rng, key = jax.random.split(rng)
    next_aug_image = batched_random_crop(key, batch.next_observations.image)
    observations = batch.observations.replace(
        image=aug_image,
    )
    next_observations = batch.next_observations.replace(
        image=next_aug_image,
    )
    batch = batch._replace(observations=observations,
                           next_observations=next_observations)

    # update critic network
    rng, key = jax.random.split(rng)
    if rc_loss_coefficient > 0:
        # reshape and normalize tactile observations
        B = batch.rewards.shape[0]
        # flatten the dimensions of the label
        tactile_label = observations.tactile.reshape(B, -1)
        new_tactile_normalizer_state = tactile_normalizer.update_stats(
            x=tactile_label, normalizer_state=tactile_normalizer_state)
        tactile_label = tactile_normalizer.normalize(tactile_label, normalizer_state=new_tactile_normalizer_state)
        new_critic, critic_info = update_critic_with_rc_loss(
            key,
            actor,
            critic,
            target_critic,
            temp,
            batch,
            discount,
            rc_loss_coefficient=rc_loss_coefficient,
            tactile_label=tactile_label,
            backup_entropy=True
        )
        if update_target:
            new_target_critic = target_update_with_decoder(new_critic, target_critic, tau)
        else:
            new_target_critic = target_critic
    else:
        new_critic, critic_info = update_critic(key,
                                                actor,
                                                critic,
                                                target_critic,
                                                temp,
                                                batch,
                                                discount,
                                                backup_entropy=True)
        new_tactile_normalizer_state = tactile_normalizer_state
        if update_target:
            new_target_critic = target_update(new_critic, target_critic, tau)
        else:
            new_target_critic = target_critic

    # Use critic conv layers in actor:
    new_actor_params = actor.params.copy()
    new_actor_params.update({'SharedEncoder': new_critic.params['SharedEncoder']})
    actor = actor.replace(params=new_actor_params)

    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, new_critic, temp, batch)
    new_temp, alpha_info = temperature.update(temp, actor_info['entropy'],
                                              target_entropy, use_log_transform=use_log_transform)

    return rng, new_actor, new_critic, new_target_critic, new_temp, \
        new_tactile_normalizer_state, {
        **critic_info,
        **actor_info,
        **alpha_info,
        'tactile_normalizer_state_mean': new_tactile_normalizer_state.mean.mean(),
        'tactile_normalizer_state_std': new_tactile_normalizer_state.std.mean(),
    }


class TactileDrQExplorer(object):

    def __init__(self,
                 seed: int,
                 observations: Dict,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 ens_lr: float = 3e-4,
                 ens_wd: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 cnn_features: Sequence[int] = (32, 32, 32, 32),
                 cnn_strides: Sequence[int] = (2, 1, 1, 1),
                 model_hidden_dims: Sequence[int] = (256, 256),
                 cnn_padding: str = 'VALID',
                 latent_dim: int = 50,
                 discount: float = 0.99,
                 tau: float = 0.005,
                 target_update_period: int = 1,
                 target_entropy: Optional[float] = None,
                 init_temperature: float = 0.1,
                 use_log_transform: bool = True,
                 switch_frequency: int = 64,
                 explore_until: int = 100_000,
                 obs_dim: int = 64,
                 num_heads: int = 5,
                 predict_reward: bool = False,
                 predict_diff: bool = False,
                 learn_std: bool = False,
                 expl_agent_update_period: int = 1,
                 ensemble_update_period: int = 1,
                 agent_update_period: int = 1,
                 max_gradient_norm: Optional[float] = None,
                 mask_touch: bool = False,
                 latent_disg_weight: float = 1.0,
                 visual_disg_weight: float = 1.0,
                 tactile_disg_weight: float = 1.0,
                 rc_loss_coefficient: float = -1,
                 ):

        self.predict_diff = predict_diff
        self.predict_reward = predict_reward
        self.expl_agent_update_period = expl_agent_update_period
        self.agent_update_period = agent_update_period
        self.ensemble_update_period = ensemble_update_period

        observations = convert_obs_to_tact_obs(observations)
        if rc_loss_coefficient > 0:
            assert observations.tactile is not None
            self.tactile_normalizer_state = NormalizerState(
                mean=jnp.zeros_like(observations.tactile[0].reshape(-1)),
                std=jnp.ones_like(observations.tactile[0].reshape(-1)),
                num_points=0,
            )
        else:
            self.tactile_normalizer_state = NormalizerState(
                mean=jnp.zeros(1),
                std=jnp.ones(1),
                num_points=0,
            )
        self.tactile_normalizer = Normalizer()

        self.rc_loss_coefficient = rc_loss_coefficient
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

        actor_optimizer = optax.adam(learning_rate=actor_lr)
        critic_optimizer = optax.adam(learning_rate=critic_lr)
        temp_optimizer = optax.adam(learning_rate=temp_lr)
        model_optimizer = optax.adamw(learning_rate=ens_lr, weight_decay=ens_wd)
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

            temp_optimizer = optax.chain(
                optax.clip_by_global_norm(max_gradient_norm),
                temp_optimizer,
            )

            model_optimizer = optax.chain(
                optax.clip_by_global_norm(max_gradient_norm),
                model_optimizer,
            )

        # use tactile drq policy with the visual tactile encoder
        actor_def = TactileDrQPolicy(hidden_dims, action_dim, cnn_features,
                                     cnn_strides, cnn_padding, latent_dim, mask_touch=mask_touch)

        actor_key, expl_actor_key = jax.random.split(actor_key)

        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=actor_optimizer)

        expl_actor = Model.create(actor_def,
                                  inputs=[expl_actor_key, observations],
                                  tx=actor_optimizer)

        # use tactile drq critic with the visual tactile encoder
        critic_def = TactileMaxInfoDrQDoubleCritic(hidden_dims, cnn_features, cnn_strides,
                                                   cnn_padding, latent_dim, obs_dim=obs_dim, mask_touch=mask_touch,
                                                   add_decoder=self.rc_loss_coefficient > 0)

        critic_key, expl_critic_key = jax.random.split(critic_key)
        critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              tx=critic_optimizer)
        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions])

        expl_critic = Model.create(critic_def,
                                   inputs=[expl_critic_key, observations, actions],
                                   tx=critic_optimizer)
        expl_target_critic = Model.create(
            critic_def, inputs=[expl_critic_key, observations, actions])

        temp_key, expl_temp_key = jax.random.split(temp_key)
        temp = Model.create(temperature.Temperature(init_temperature),
                            inputs=[temp_key],
                            tx=temp_optimizer)

        expl_temp = Model.create(temperature.Temperature(init_temperature),
                                 inputs=[expl_temp_key],
                                 tx=temp_optimizer)

        model_key, rng = jax.random.split(rng, 2)

        # add an additional dimension for the tactile observation
        # add observation dim if tactile obs is not none
        output_dim = latent_dim + obs_dim * (1 + int(observations.tactile is not None))
        if predict_reward:
            output_dim += 1
        disg_weights = jnp.ones(output_dim)
        disg_weights = disg_weights.at[:latent_dim].set(latent_disg_weight)
        disg_weights = disg_weights.at[latent_dim: latent_dim + obs_dim].set(visual_disg_weight)
        if observations.tactile is not None:
            disg_weights = disg_weights.at[latent_dim + obs_dim: latent_dim + 2 * obs_dim].set(tactile_disg_weight)

        if learn_std:
            model_type = ProbabilisticEnsemble
        else:
            model_type = DeterministicEnsemble
        ensemble = model_type(
            model_kwargs={'hidden_dims': model_hidden_dims + (output_dim,),
                          'init_scale': jnp.sqrt(2)},
            optimizer=model_optimizer,
            num_heads=num_heads)

        dummy_state = jnp.zeros((actions.shape[0], latent_dim))
        ens_state = ensemble.init(key=model_key, input=jnp.concatenate([dummy_state, actions], axis=-1))
        ensemble.set_disg_weights(disg_weights=disg_weights)

        self.actor = actor
        self.expl_actor = expl_actor
        self.critic = critic
        self.expl_critic = expl_critic
        self.target_critic = target_critic
        self.expl_target_critic = expl_target_critic
        self.temp = temp
        self.expl_temp = expl_temp
        self.rng = rng
        self.step = 0
        self.expl_step = 0
        self.use_log_transform = use_log_transform
        self.exploring = False
        self.switch_frequency = switch_frequency
        self.explore_until = explore_until

        self.ens = ensemble
        self.ens_state = ens_state

    @property
    def switch_strategy(self):
        return self.step % self.switch_frequency == 0

    def sample_actions(self,
                       observations: Dict,
                       temperature: float = 1.0) -> np.ndarray:

        # convert obs to tactile obs
        tact_obs = convert_obs_to_tact_obs(observations)

        # we are in evaluation mode
        if temperature == 0:
            rng, actions = policies.sample_actions(self.rng, self.actor.apply_fn,
                                                   self.actor.params, tact_obs,
                                                   temperature)

        else:
            if self.step <= self.explore_until:
                if self.switch_strategy:
                    if self.exploring:
                        self.exploring = False
                    else:
                        self.exploring = True
            else:
                self.exploring = False

            if self.exploring:
                self.expl_step += 1
                rng, actions = policies.sample_actions(self.rng, self.expl_actor.apply_fn,
                                                       self.expl_actor.params, tact_obs,
                                                       temperature)
            else:
                rng, actions = policies.sample_actions(self.rng, self.actor.apply_fn,
                                                       self.actor.params, tact_obs,
                                                       temperature)

        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch) -> InfoDict:
        obs, next_obs = convert_obs_to_tact_obs(batch.observations, batch.next_observations)
        batch = batch._replace(observations=obs, next_observations=next_obs)
        self.step += 1
        info = {}

        new_ens_state, expl_batch = get_expl_batch(
            batch=batch,
            target_critic=self.target_critic,
            ens=self.ens,
            ens_state=self.ens_state,
        )
        self.ens_state = new_ens_state

        if self.step % self.ensemble_update_period == 0:
            new_ens_state, ens_info = update_ensemble(batch=batch,
                                                      target_critic=self.target_critic,
                                                      ens=self.ens,
                                                      ens_state=self.ens_state,
                                                      predict_diff=self.predict_diff,
                                                      predict_rewards=self.predict_reward,
                                                      )

            self.ens_state = new_ens_state

            info = ens_info

        if self.step % self.agent_update_period == 0:
            new_rng, new_actor, new_critic, \
                new_target_critic, new_temp, new_tactile_normalizer_state, agent_info = _update_jit(
                self.rng, self.actor, self.critic, self.target_critic, self.temp,
                batch, self.discount, self.tau, self.target_entropy,
                self.step % self.target_update_period == 0,
                tactile_normalizer=self.tactile_normalizer,
                rc_loss_coefficient=self.rc_loss_coefficient,
                tactile_normalizer_state=self.tactile_normalizer_state,
                use_log_transform=self.use_log_transform)

            self.rng = new_rng
            self.actor = new_actor
            self.critic = new_critic
            self.target_critic = new_target_critic
            self.temp = new_temp
            self.tactile_normalizer_state = new_tactile_normalizer_state

            info = info | agent_info

        if self.step % self.expl_agent_update_period == 0:
            new_rng, new_actor, new_critic, \
                new_target_critic, new_temp, new_tactile_normalizer_state, agent_info = _update_jit(
                self.rng, self.expl_actor, self.expl_critic, self.expl_target_critic, self.expl_temp,
                expl_batch, self.discount, self.tau, self.target_entropy,
                self.step % self.target_update_period == 0,
                tactile_normalizer=self.tactile_normalizer,
                rc_loss_coefficient=self.rc_loss_coefficient,
                tactile_normalizer_state=self.tactile_normalizer_state,
                use_log_transform=self.use_log_transform)

            self.rng = new_rng
            self.expl_actor = new_actor
            self.expl_critic = new_critic
            self.expl_target_critic = new_target_critic
            self.expl_temp = new_temp
            self.tactile_normalizer_state = new_tactile_normalizer_state
            agent_info = {f'expl_{k}': v for k, v in agent_info.items()}
            info = info | agent_info

        return info | {'policy_phase': int(self.exploring), 'exploration_steps': self.expl_step}
