"""On-policy PPO learner for diffusion-steering with pixel observations."""

import functools
import os
import pathlib
import re
import tempfile
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core.frozen_dict import FrozenDict
from flax import serialization
from flax.training import train_state

from jaxrl2.agents.agent import Agent
from jaxrl2.agents.pixel_maxinfosac.networks import PixelTactileMultiplexer
from jaxrl2.agents.pixel_ppo.networks import (
    StateIndependentNormalPolicy,
    sample_squashed_action_and_log_prob,
    squash_action,
    squashed_log_prob,
)
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.networks.encoders.impala_encoder import (
    ImpalaEncoder,
    SmallerImpalaEncoder,
)
from jaxrl2.networks.encoders.networks import Encoder, PixelMultiplexer
from jaxrl2.networks.encoders.resnet_encoderv1 import (
    ResNet18,
    ResNet34,
    ResNetSmall,
)
from jaxrl2.networks.encoders.resnet_encoderv2 import ResNetV2Encoder
from jaxrl2.networks.values import StateValue


class TrainState(train_state.TrainState):
    batch_stats: Any


def _model_variables(state: TrainState, params) -> Dict[str, Any]:
    variables = {"params": params}
    if state.batch_stats is not None:
        variables["batch_stats"] = state.batch_stats
    return variables


def _apply_actor(actor: TrainState, params, observations):
    return actor.apply_fn(
        _model_variables(actor, params),
        observations,
        training=False,
        mutable=False,
    )


def _apply_value(value_state: TrainState, params, observations):
    return value_state.apply_fn(
        _model_variables(value_state, params),
        observations,
        training=False,
        mutable=False,
    )


def clipped_policy_loss(
        new_log_probs: jnp.ndarray,
        old_log_probs: jnp.ndarray,
        advantages: jnp.ndarray,
        clip_epsilon: float,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Computes PPO's clipped surrogate objective and ratio diagnostics."""

    log_ratio = new_log_probs - old_log_probs
    # Extreme stale batches should not turn a diagnostic into inf/NaN. PPO still
    # sees a fully saturated clipped objective at either end of this interval.
    ratio = jnp.exp(jnp.clip(log_ratio, -20.0, 20.0))
    unclipped = ratio * advantages
    clipped_ratio = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    clipped = clipped_ratio * advantages
    loss = -jnp.mean(jnp.minimum(unclipped, clipped))
    approx_kl = jnp.mean((ratio - 1.0) - log_ratio)
    clip_fraction = jnp.mean(
        (jnp.abs(ratio - 1.0) > clip_epsilon).astype(jnp.float32)
    )
    return loss, {
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
        "ratio": jnp.mean(ratio),
    }


def clipped_value_loss(
        values: jnp.ndarray,
        old_values: jnp.ndarray,
        returns: jnp.ndarray,
        clip_epsilon: float,
) -> jnp.ndarray:
    """Computes the clipped PPO value loss."""

    clipped_values = old_values + jnp.clip(
        values - old_values,
        -clip_epsilon,
        clip_epsilon,
    )
    value_losses = jnp.square(values - returns)
    clipped_value_losses = jnp.square(clipped_values - returns)
    return 0.5 * jnp.mean(jnp.maximum(value_losses, clipped_value_losses))


@functools.partial(jax.jit, static_argnames=())
def _sample_actions_with_info_jit(
        rng,
        actor: TrainState,
        value_state: TrainState,
        observations,
        action_magnitude: float,
):
    rng, sample_key = jax.random.split(rng)
    distribution = _apply_actor(actor, actor.params, observations)
    action, pre_tanh_action, log_prob = sample_squashed_action_and_log_prob(
        distribution,
        sample_key,
        action_magnitude,
    )
    values = _apply_value(value_state, value_state.params, observations)
    return rng, action, pre_tanh_action, log_prob, values


@functools.partial(jax.jit, static_argnames=())
def _eval_actions_jit(
        actor: TrainState,
        observations,
        action_magnitude: float,
):
    distribution = _apply_actor(actor, actor.params, observations)
    return squash_action(distribution.mode(), action_magnitude)


@functools.partial(jax.jit, static_argnames=())
def _value_jit(value_state: TrainState, observations):
    return _apply_value(value_state, value_state.params, observations)


@functools.partial(jax.jit, static_argnames=())
def _update_minibatch_jit(
        rng,
        actor: TrainState,
        value_state: TrainState,
        batch,
        action_magnitude: float,
        clip_epsilon: float,
        value_loss_coefficient: float,
        entropy_coefficient: float,
):
    rng, entropy_key = jax.random.split(rng)
    batch_size = batch["old_log_probs"].shape[0]
    pre_tanh_actions = jnp.reshape(
        batch["pre_tanh_actions"],
        (batch_size, -1),
    )
    old_log_probs = jnp.reshape(batch["old_log_probs"], (-1,))
    old_values = jnp.reshape(batch["old_values"], (-1,))
    advantages = jnp.reshape(batch["advantages"], (-1,))
    returns = jnp.reshape(batch["returns"], (-1,))

    def actor_loss_fn(actor_params):
        distribution = _apply_actor(actor, actor_params, batch["observations"])
        new_log_probs = squashed_log_prob(
            distribution,
            pre_tanh_actions,
            action_magnitude,
        )
        policy_loss, policy_info = clipped_policy_loss(
            new_log_probs,
            old_log_probs,
            advantages,
            clip_epsilon,
        )

        entropy_pre_tanh = distribution.sample(seed=entropy_key)
        entropy = -jnp.mean(squashed_log_prob(
            distribution,
            entropy_pre_tanh,
            action_magnitude,
        ))
        actor_loss = policy_loss - entropy_coefficient * entropy
        return actor_loss, {
            "actor_loss": actor_loss,
            "policy_loss": policy_loss,
            "entropy": entropy,
            "new_log_prob": jnp.mean(new_log_probs),
            **policy_info,
        }

    (actor_loss, actor_info), actor_grads = jax.value_and_grad(
        actor_loss_fn,
        has_aux=True,
    )(actor.params)
    actor_grad_norm = optax.global_norm(actor_grads)
    new_actor = actor.apply_gradients(grads=actor_grads)

    def value_loss_fn(value_params):
        values = _apply_value(
            value_state,
            value_params,
            batch["observations"],
        )
        values = jnp.reshape(values, (-1,))
        unscaled_value_loss = clipped_value_loss(
            values,
            old_values,
            returns,
            clip_epsilon,
        )
        scaled_value_loss = value_loss_coefficient * unscaled_value_loss
        return scaled_value_loss, {
            "value_loss": unscaled_value_loss,
            "value": jnp.mean(values),
            "return": jnp.mean(returns),
        }

    (scaled_value_loss, value_info), value_grads = jax.value_and_grad(
        value_loss_fn,
        has_aux=True,
    )(value_state.params)
    value_grad_norm = optax.global_norm(value_grads)
    new_value_state = value_state.apply_gradients(grads=value_grads)

    total_grad_norm = jnp.sqrt(
        jnp.square(actor_grad_norm) + jnp.square(value_grad_norm)
    )
    info = {
        **actor_info,
        **value_info,
        "loss": actor_loss + scaled_value_loss,
        "actor_grad_norm": actor_grad_norm,
        "value_grad_norm": value_grad_norm,
        "grad_norm": total_grad_norm,
    }
    return rng, new_actor, new_value_state, info


def _make_encoder(
        encoder_type: str,
        cnn_features: Sequence[int],
        cnn_strides: Sequence[int],
        cnn_padding: str,
        encoder_norm: str,
        use_spatial_softmax: bool,
        softmax_temperature: float,
):
    if encoder_type == "small":
        return Encoder(cnn_features, cnn_strides, cnn_padding)
    if encoder_type == "impala":
        return ImpalaEncoder()
    if encoder_type == "impala_small":
        return SmallerImpalaEncoder()
    if encoder_type == "resnet_small":
        return ResNetSmall(
            norm=encoder_norm,
            use_spatial_softmax=use_spatial_softmax,
            softmax_temperature=softmax_temperature,
        )
    if encoder_type == "resnet_18_v1":
        return ResNet18(
            norm=encoder_norm,
            use_spatial_softmax=use_spatial_softmax,
            softmax_temperature=softmax_temperature,
        )
    if encoder_type == "resnet_34_v1":
        return ResNet34(
            norm=encoder_norm,
            use_spatial_softmax=use_spatial_softmax,
            softmax_temperature=softmax_temperature,
        )
    if encoder_type == "resnet_small_v2":
        return ResNetV2Encoder(stage_sizes=(1, 1, 1, 1), norm=encoder_norm)
    if encoder_type == "resnet_18_v2":
        return ResNetV2Encoder(stage_sizes=(2, 2, 2, 2), norm=encoder_norm)
    if encoder_type == "resnet_34_v2":
        return ResNetV2Encoder(stage_sizes=(3, 4, 6, 3), norm=encoder_norm)
    raise ValueError(f"Unknown encoder type: {encoder_type}")


def _make_pixel_network(
        encoder,
        network,
        has_tactile: bool,
        latent_dim: int,
        use_bottleneck: bool,
        obs_dim: int,
        tactile_hidden_dims: Sequence[int],
        cnn_features: Sequence[int],
        cnn_strides: Sequence[int],
        cnn_padding: str,
        mask_touch: bool,
):
    if has_tactile:
        return PixelTactileMultiplexer(
            encoder=encoder,
            network=network,
            latent_dim=latent_dim,
            use_bottleneck=use_bottleneck,
            obs_dim=obs_dim,
            tactile_hidden_dims=tactile_hidden_dims,
            tactile_cnn_features=cnn_features,
            tactile_cnn_strides=cnn_strides,
            tactile_cnn_padding=cnn_padding,
            mask_touch=mask_touch,
        )
    return PixelMultiplexer(
        encoder=encoder,
        network=network,
        latent_dim=latent_dim,
        use_bottleneck=use_bottleneck,
    )


class PixelPPOLearner(Agent):
    """PPO over the bounded query-level action used to steer pi-zero."""

    def __init__(
            self,
            seed: int,
            observations: Union[jnp.ndarray, DatasetDict],
            actions: jnp.ndarray,
            ppo_lr: float = 3e-4,
            ppo_rollout_steps: int = 256,
            ppo_num_epochs: int = 10,
            ppo_num_minibatches: int = 4,
            ppo_clip_epsilon: float = 0.2,
            ppo_gae_lambda: float = 0.95,
            ppo_value_loss_coef: float = 0.5,
            ppo_entropy_coef: float = 1e-2,
            ppo_max_grad_norm: float = 0.5,
            ppo_target_kl: float = -1.0,
            ppo_init_log_std: float = 0.0,
            ppo_normalize_advantages: bool = True,
            hidden_dims: Sequence[int] = (256, 256),
            cnn_features: Sequence[int] = (32, 32, 32, 32),
            cnn_strides: Sequence[int] = (2, 1, 1, 1),
            cnn_padding: str = "VALID",
            latent_dim: int = 50,
            dropout_rate: Optional[float] = None,
            encoder_type: str = "resnet_34_v1",
            encoder_norm: str = "group",
            use_spatial_softmax: bool = True,
            softmax_temperature: float = 1.0,
            use_bottleneck: bool = True,
            obs_dim: int = 64,
            tactile_hidden_dims: Sequence[int] = (256, 256),
            mask_touch: bool = False,
            action_magnitude: float = 1.0,
            discount: float = 0.999,
            num_cameras: int = 1,
    ):
        if ppo_lr <= 0:
            raise ValueError("ppo_lr must be positive.")
        if ppo_rollout_steps <= 0:
            raise ValueError("ppo_rollout_steps must be positive.")
        if ppo_num_epochs <= 0:
            raise ValueError("ppo_num_epochs must be positive.")
        if ppo_num_minibatches <= 0:
            raise ValueError("ppo_num_minibatches must be positive.")
        if not 0.0 <= ppo_clip_epsilon < 1.0:
            raise ValueError("ppo_clip_epsilon must be in [0, 1).")
        if not 0.0 <= ppo_gae_lambda <= 1.0:
            raise ValueError("ppo_gae_lambda must be in [0, 1].")
        if ppo_value_loss_coef < 0 or ppo_entropy_coef < 0:
            raise ValueError("PPO loss coefficients must be non-negative.")
        if ppo_max_grad_norm <= 0:
            raise ValueError("ppo_max_grad_norm must be positive.")
        if action_magnitude <= 0:
            raise ValueError("action_magnitude must be positive.")

        self.action_dim = int(np.prod(actions.shape[-2:]))
        self.action_chunk_shape = tuple(actions.shape[-2:])
        self.action_magnitude = float(action_magnitude)
        self.discount = float(discount)
        self.num_cameras = int(num_cameras)

        self.ppo_rollout_steps = int(ppo_rollout_steps)
        self.ppo_num_epochs = int(ppo_num_epochs)
        self.ppo_num_minibatches = int(ppo_num_minibatches)
        self.ppo_clip_epsilon = float(ppo_clip_epsilon)
        self.ppo_gae_lambda = float(ppo_gae_lambda)
        self.ppo_value_loss_coef = float(ppo_value_loss_coef)
        self.ppo_entropy_coef = float(ppo_entropy_coef)
        self.ppo_target_kl = (
            None if ppo_target_kl is None or float(ppo_target_kl) <= 0
            else float(ppo_target_kl)
        )
        self.ppo_normalize_advantages = bool(ppo_normalize_advantages)
        self._num_updates = 0

        if len(hidden_dims) == 1:
            hidden_dims = (hidden_dims[0], hidden_dims[0], hidden_dims[0])
        hidden_dims = tuple(hidden_dims)

        has_tactile = (
            isinstance(observations, (dict, FrozenDict))
            and "tactile" in observations
        )
        actor_encoder = _make_encoder(
            encoder_type,
            cnn_features,
            cnn_strides,
            cnn_padding,
            encoder_norm,
            use_spatial_softmax,
            softmax_temperature,
        )
        value_encoder = _make_encoder(
            encoder_type,
            cnn_features,
            cnn_strides,
            cnn_padding,
            encoder_norm,
            use_spatial_softmax,
            softmax_temperature,
        )
        policy = StateIndependentNormalPolicy(
            hidden_dims=hidden_dims,
            action_dim=self.action_dim,
            dropout_rate=dropout_rate,
            init_log_std=ppo_init_log_std,
        )
        value_head = StateValue(hidden_dims=hidden_dims)
        actor_definition = _make_pixel_network(
            actor_encoder,
            policy,
            has_tactile,
            latent_dim,
            use_bottleneck,
            obs_dim,
            tactile_hidden_dims,
            cnn_features,
            cnn_strides,
            cnn_padding,
            mask_touch,
        )
        value_definition = _make_pixel_network(
            value_encoder,
            value_head,
            has_tactile,
            latent_dim,
            use_bottleneck,
            obs_dim,
            tactile_hidden_dims,
            cnn_features,
            cnn_strides,
            cnn_padding,
            mask_touch,
        )

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, value_key = jax.random.split(rng, 3)
        actor_variables = actor_definition.init(actor_key, observations)
        value_variables = value_definition.init(value_key, observations)

        optimizer = optax.chain(
            optax.clip_by_global_norm(float(ppo_max_grad_norm)),
            optax.adam(learning_rate=float(ppo_lr)),
        )
        self._actor = TrainState.create(
            apply_fn=actor_definition.apply,
            params=actor_variables["params"],
            tx=optimizer,
            batch_stats=actor_variables.get("batch_stats"),
        )
        self._value_state = TrainState.create(
            apply_fn=value_definition.apply,
            params=value_variables["params"],
            tx=optimizer,
            batch_stats=value_variables.get("batch_stats"),
        )
        self._rng = rng
        # Evaluation must sample diffusion noise, but it must not perturb the
        # on-policy behavior RNG used by subsequent training rollouts.
        self._eval_rng = jax.random.fold_in(rng, 0x50504F)

    def sample_actions_with_info(self, observations) -> Dict[str, np.ndarray]:
        """Samples one on-policy action and records ratio/value statistics."""

        rng, action, pre_tanh_action, log_prob, values = (
            _sample_actions_with_info_jit(
                self._rng,
                self._actor,
                self._value_state,
                observations,
                self.action_magnitude,
            )
        )
        self._rng = rng
        return {
            "action": np.asarray(action),
            "pre_tanh_action": np.asarray(pre_tanh_action),
            "log_prob": np.asarray(log_prob),
            "value": np.asarray(values),
        }

    def sample_actions(self, observations) -> np.ndarray:
        return self.sample_actions_with_info(observations)["action"]

    def sample_eval_actions(self, observations) -> np.ndarray:
        rng, action, _, _, _ = _sample_actions_with_info_jit(
            self._eval_rng,
            self._actor,
            self._value_state,
            observations,
            self.action_magnitude,
        )
        self._eval_rng = rng
        return np.asarray(action)

    def eval_actions(self, observations) -> np.ndarray:
        actions = _eval_actions_jit(
            self._actor,
            observations,
            self.action_magnitude,
        )
        return np.asarray(actions)

    def value(self, observations) -> np.ndarray:
        return np.asarray(_value_jit(self._value_state, observations))

    def update(self, batch) -> Dict[str, jnp.ndarray]:
        """Runs all configured PPO epochs over one fresh on-policy rollout."""

        required_keys = {
            "observations",
            "actions",
            "pre_tanh_actions",
            "old_log_probs",
            "old_values",
            "advantages",
            "returns",
        }
        missing = required_keys.difference(batch.keys())
        if missing:
            raise KeyError(f"PPO batch is missing keys: {sorted(missing)}")

        batch_size = int(np.shape(batch["old_log_probs"])[0])
        if batch_size <= 0:
            raise ValueError("PPO cannot update from an empty rollout.")

        prepared_batch = dict(batch)
        advantages = jnp.asarray(prepared_batch["advantages"])
        if self.ppo_normalize_advantages:
            advantages = (
                advantages - jnp.mean(advantages)
            ) / (jnp.std(advantages) + 1e-8)
        prepared_batch["advantages"] = advantages

        num_minibatches = min(self.ppo_num_minibatches, batch_size)
        all_metrics = []
        epochs_completed = 0
        early_stopped = False

        for _ in range(self.ppo_num_epochs):
            self._rng, permutation_key = jax.random.split(self._rng)
            permutation = np.asarray(
                jax.random.permutation(permutation_key, batch_size)
            )
            epoch_kls = []
            for indices in np.array_split(permutation, num_minibatches):
                minibatch = jax.tree_util.tree_map(
                    lambda value: jnp.asarray(value)[indices],
                    prepared_batch,
                )
                self._rng, self._actor, self._value_state, update_info = (
                    _update_minibatch_jit(
                        self._rng,
                        self._actor,
                        self._value_state,
                        minibatch,
                        self.action_magnitude,
                        self.ppo_clip_epsilon,
                        self.ppo_value_loss_coef,
                        self.ppo_entropy_coef,
                    )
                )
                host_info = jax.device_get(update_info)
                all_metrics.append(host_info)
                epoch_kls.append(float(host_info["approx_kl"]))

            epochs_completed += 1
            if (
                    self.ppo_target_kl is not None
                    and np.mean(epoch_kls) > self.ppo_target_kl):
                early_stopped = True
                break

        self._num_updates += 1
        mean_metrics = {
            key: jnp.asarray(np.mean([float(info[key]) for info in all_metrics]))
            for key in all_metrics[0]
        }

        returns = np.asarray(prepared_batch["returns"]).reshape(-1)
        updated_values = self.value(prepared_batch["observations"]).reshape(-1)
        return_variance = float(np.var(returns))
        if return_variance < 1e-8:
            explained_variance = 0.0
        else:
            explained_variance = 1.0 - float(
                np.var(returns - updated_values) / return_variance
            )
        mean_metrics.update({
            "explained_variance": jnp.asarray(explained_variance),
            "epochs_completed": jnp.asarray(epochs_completed),
            "minibatches_completed": jnp.asarray(len(all_metrics)),
            "early_stop": jnp.asarray(float(early_stopped)),
        })
        return mean_metrics

    @property
    def _save_dict(self):
        return {
            "actor": self._actor,
            "value": self._value_state,
            "rng": self._rng,
            "num_updates": self._num_updates,
        }

    def save_checkpoint(self, directory, step, keep_every_n_steps=None):
        """Synchronously writes an atomic PPO checkpoint.

        The repository's inherited checkpoint helper delegates to an async
        Orbax path in the pinned Flax version.  PPO uses a small synchronous
        msgpack file instead so collection cannot race an unfinished write.
        Existing checkpoints are never rotated or removed.
        """

        del keep_every_n_steps  # PPO intentionally keeps every requested save.
        directory = pathlib.Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint_path = directory / f"checkpoint{int(step)}"
        payload = serialization.to_bytes(self._save_dict)

        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=str(directory),
            prefix=f".{checkpoint_path.name}.",
            suffix=".tmp",
        )
        temporary_path = pathlib.Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "wb") as checkpoint_file:
                checkpoint_file.write(payload)
                checkpoint_file.flush()
                os.fsync(checkpoint_file.fileno())
            os.replace(temporary_path, checkpoint_path)
        except BaseException:
            try:
                temporary_path.unlink(missing_ok=True)
            finally:
                raise
        return str(checkpoint_path)

    def restore_checkpoint(self, directory):
        directory = pathlib.Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Checkpoint {directory} does not exist.")
        if not directory.is_dir():
            raise NotADirectoryError(
                f"PPO checkpoint path must be a directory: {directory}"
            )

        checkpoint_pattern = re.compile(r"^checkpoint(?P<step>\d+)$")
        candidates = []
        for path in directory.iterdir():
            match = checkpoint_pattern.fullmatch(path.name)
            if match is not None and path.is_file():
                candidates.append((int(match.group("step")), path))
        if not candidates:
            raise FileNotFoundError(
                f"No numeric PPO checkpoints found in {directory}."
            )

        _, checkpoint_path = max(candidates, key=lambda item: item[0])
        restored = serialization.from_bytes(
            self._save_dict,
            checkpoint_path.read_bytes(),
        )
        self._actor = restored["actor"]
        self._value_state = restored["value"]
        self._rng = restored.get("rng", self._rng)
        self._num_updates = int(restored.get("num_updates", self._num_updates))
        print("restored from ", checkpoint_path)


__all__ = (
    "PixelPPOLearner",
    "clipped_policy_loss",
    "clipped_value_loss",
)
