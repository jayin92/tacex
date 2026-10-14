"""Deterministic ensemble-disagreement rewards for DPPO.

This module intentionally contains no policy or training-loop code.  It turns
raw DPPO observations into fixed-size, deterministic features and applies the
same dynamics-ensemble disagreement used by the existing PixelMaxInfoSAC
agents.  The ensemble state is threaded explicitly so an on-policy caller can
score a rollout with the pre-update model before fitting that model.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import jax
import jax.numpy as jnp
import optax

from jaxrl2.agents.pixel_maxinfosac.ensemble_utils import (
    disagreement_weights as make_disagreement_weights,
    ensemble_inputs,
    ensemble_targets_from_modalities,
    modality_output_slices,
)
from jaxrl2.networks.ensemble_model import (
    DeterministicEnsemble,
    EnsembleState,
)


MODALITY_ORDER = ("image", "state", "tactile")
_OBSERVATION_KEY = {
    "image": "pixels",
    "state": "state",
    "tactile": "tactile",
}


@functools.partial(jax.jit, static_argnums=(0,))
def _update_ensemble_params(ensemble, model_input, model_output, state):
    """Fits params using existing normalizers without recounting the batch."""
    normalizers = state.ensemble_normalizer_state
    normalized_input = ensemble.normalizer.normalize(
        model_input, normalizers.input_normalizer_state)
    normalized_output = ensemble.normalizer.normalize(
        model_output, normalizers.output_normalizer_state)
    (loss, mse), grads = jax.value_and_grad(
        ensemble.loss, has_aux=True)(
            state.vmapped_params, normalized_input, normalized_output)
    updates, new_opt_state = ensemble.tx.update(
        grads, state.opt_state, state.vmapped_params)
    new_params = optax.apply_updates(state.vmapped_params, updates)
    return state.replace(
        vmapped_params=new_params,
        opt_state=new_opt_state,
        step=state.step + 1,
    ), (loss, mse)


def _max_pool_to_dim(value: Any, obs_dim: int, *, name: str) -> jax.Array:
    """Flattens a batch and deterministically max-pools it to ``obs_dim``."""

    value = jnp.asarray(value, dtype=jnp.float32)
    if value.ndim < 2:
        raise ValueError(
            f"Observation modality {name!r} must have a leading batch "
            f"dimension; got shape {value.shape}.")

    value = jnp.reshape(value, (value.shape[0], -1))
    feature_dim = value.shape[-1]
    if feature_dim == 0:
        raise ValueError(f"Observation modality {name!r} is empty.")
    if feature_dim == obs_dim:
        return value
    if feature_dim < obs_dim:
        return jnp.pad(value, ((0, 0), (0, obs_dim - feature_dim)))

    pool_size = (feature_dim + obs_dim - 1) // obs_dim
    padded_dim = pool_size * obs_dim
    if padded_dim != feature_dim:
        value = jnp.pad(
            value,
            ((0, 0), (0, padded_dim - feature_dim)),
            constant_values=-jnp.inf,
        )
    value = jnp.reshape(value, (value.shape[0], obs_dim, pool_size))
    return jnp.max(value, axis=-1)


def modality_features(
        observations: Mapping[str, Any],
        obs_dim: int,
) -> dict[str, jax.Array]:
    """Builds ordered, fixed-size features from a batch of observations.

    ``pixels`` must be uint8 and is scaled to ``[0, 1]`` before pooling.
    State and tactile values are converted to float32 without rescaling.  Only
    modalities present in ``observations`` are returned, in image/state/tactile
    order; the image feature is sourced from the repository's ``pixels`` key.
    """

    if not isinstance(observations, Mapping):
        raise TypeError("observations must be a mapping of modality arrays.")
    obs_dim = int(obs_dim)
    if obs_dim <= 0:
        raise ValueError(f"obs_dim must be positive, got {obs_dim}.")

    features: dict[str, jax.Array] = {}
    batch_size: Optional[int] = None
    for modality in MODALITY_ORDER:
        observation_key = _OBSERVATION_KEY[modality]
        if observation_key not in observations:
            continue

        value = jnp.asarray(observations[observation_key])
        if modality == "image":
            if value.dtype != jnp.uint8:
                raise TypeError(
                    "observations['pixels'] must have dtype uint8 so image "
                    "scaling is unambiguous; got "
                    f"{value.dtype}.")
            value = value.astype(jnp.float32) / jnp.float32(255.0)
        else:
            value = value.astype(jnp.float32)

        feature = _max_pool_to_dim(value, obs_dim, name=modality)
        if batch_size is None:
            batch_size = feature.shape[0]
        elif feature.shape[0] != batch_size:
            raise ValueError(
                "All observation modalities must have the same batch size; "
                f"expected {batch_size}, got {feature.shape[0]} for "
                f"{modality!r}.")
        features[modality] = feature

    if not features:
        expected = tuple(_OBSERVATION_KEY.values())
        raise ValueError(
            "No supported observation modalities were provided; expected at "
            f"least one of {expected}.")
    return features


def _selected_modalities(
        selected: Optional[Sequence[str] | str],
) -> Optional[tuple[str, ...]]:
    if selected is None:
        return None
    if isinstance(selected, str):
        selected = tuple(
            item.strip() for item in selected.split(",") if item.strip())
        return selected or None
    return tuple(str(item).strip() for item in selected if str(item).strip())


def _concatenate_features(
        features: Mapping[str, jax.Array],
        modalities: Sequence[str],
) -> jax.Array:
    return jnp.concatenate([features[name] for name in modalities], axis=-1)


class DPPODisagreementReward:
    """Functional deterministic-ensemble reward used by isolated DPPO runs.

    Callers should pass the state returned by one ``score_and_update`` call to
    the next call.  Omitting ``state`` intentionally starts from
    ``initial_state`` and is useful for the first rollout and focused tests;
    this object never mutates hidden training state.
    """

    def __init__(
            self,
            seed: int,
            sample_observations: Mapping[str, Any],
            sample_actions: Any,
            *,
            obs_dim: int = 64,
            model_hidden_dims: Sequence[int] = (256, 256),
            num_model_heads: int = 5,
            model_lr: float = 3e-4,
            model_wd: float = 0.0,
            num_update_epochs: int = 1,
            predict_diff: bool = True,
            predict_reward: bool = True,
            ensemble_disagreement_modalities: Optional[
                Sequence[str] | str
            ] = None,
    ):
        self.obs_dim = int(obs_dim)
        self.predict_diff = bool(predict_diff)
        self.predict_reward = bool(predict_reward)
        self.num_update_epochs = int(num_update_epochs)
        if self.num_update_epochs <= 0:
            raise ValueError("num_update_epochs must be positive.")

        sample_features = modality_features(sample_observations, self.obs_dim)
        self.output_modalities = tuple(
            name for name in MODALITY_ORDER if name in sample_features)
        self.output_slices = modality_output_slices(
            sample_features,
            self.output_modalities,
            predict_reward=self.predict_reward,
        )

        selected_modalities = _selected_modalities(
            ensemble_disagreement_modalities)
        self.selected_disagreement_modalities = (
            tuple(name for name in self.output_modalities)
            if selected_modalities is None
            else selected_modalities
        )
        self.disagreement_weights = make_disagreement_weights(
            self.output_slices,
            selected_modalities,
        )

        sample_actions = jnp.asarray(sample_actions, dtype=jnp.float32)
        if sample_actions.ndim < 2:
            raise ValueError(
                "sample_actions must have a leading batch dimension and at "
                f"least one action dimension; got {sample_actions.shape}.")
        if sample_actions.shape[0] != next(iter(sample_features.values())).shape[0]:
            raise ValueError(
                "sample_actions and sample_observations must have the same "
                "batch size.")
        self.action_chunk_shape = tuple(sample_actions.shape[1:])

        model_output_dim = max(end for _, end in self.output_slices.values())
        optimizer = optax.adamw(
            learning_rate=float(model_lr),
            weight_decay=float(model_wd),
        )
        self.ensemble = DeterministicEnsemble(
            model_kwargs={
                "hidden_dims": tuple(model_hidden_dims) + (model_output_dim,),
            },
            optimizer=optimizer,
            num_heads=int(num_model_heads),
        )
        sample_model_state = _concatenate_features(
            sample_features, self.output_modalities)
        sample_model_input = ensemble_inputs(sample_model_state, sample_actions)
        self.initial_state = self.ensemble.init(
            key=jax.random.PRNGKey(int(seed)),
            input=sample_model_input,
        )
        # ``init`` resets the ensemble's Python-side weight vector, so modality
        # selection must be installed after state initialization.
        self.ensemble.set_disg_weights(self.disagreement_weights)

    def _features_for_model(
            self,
            observations: Mapping[str, Any],
    ) -> dict[str, jax.Array]:
        features = modality_features(observations, self.obs_dim)
        missing = tuple(
            name for name in self.output_modalities if name not in features)
        if missing:
            observation_keys = tuple(_OBSERVATION_KEY[name] for name in missing)
            raise ValueError(
                "Observation batch is missing modalities required by the "
                f"ensemble: {missing} (observation keys {observation_keys}).")
        return features

    def score_and_update(
            self,
            observations: Mapping[str, Any],
            actions: Any,
            next_observations: Mapping[str, Any],
            extrinsic_rewards: Any,
            *,
            state: Optional[EnsembleState] = None,
    ) -> tuple[jax.Array, EnsembleState, dict[str, jax.Array]]:
        """Scores pre-update disagreement, then fits the ensemble once.

        Args:
            observations: Current batched observation dictionary.
            actions: Executed action chunks with the same non-batch shape used
                at initialization.
            next_observations: Next batched observation dictionary.
            extrinsic_rewards: Environment rewards.  These are model targets
                only when ``predict_reward`` is enabled; disagreement never
                selects the reward output.
            state: Explicit ensemble state.  Defaults to ``initial_state``.

        Returns:
            A tuple of intrinsic rewards, the post-fit ensemble state, and JAX
            scalar metrics.
        """

        if state is None:
            state = self.initial_state

        features = self._features_for_model(observations)
        next_features = self._features_for_model(next_observations)
        batch_size = next(iter(features.values())).shape[0]

        actions = jnp.asarray(actions, dtype=jnp.float32)
        expected_action_shape = (batch_size, *self.action_chunk_shape)
        if actions.shape != expected_action_shape:
            raise ValueError(
                "Executed action chunks must retain their fixed shape; "
                f"expected {expected_action_shape}, got {actions.shape}.")
        if any(value.shape[0] != batch_size for value in next_features.values()):
            raise ValueError(
                "Current and next observations must have the same batch size.")

        extrinsic_rewards = jnp.asarray(
            extrinsic_rewards, dtype=jnp.float32)
        if extrinsic_rewards.size != batch_size:
            raise ValueError(
                "extrinsic_rewards must contain exactly one value per "
                f"transition; expected {batch_size}, got "
                f"{extrinsic_rewards.size}.")
        extrinsic_rewards = jnp.reshape(extrinsic_rewards, (batch_size,))

        model_state = _concatenate_features(features, self.output_modalities)
        model_input = ensemble_inputs(model_state, actions)

        # This ordering is intentional: rollout rewards come from the ensemble
        # parameters that existed before learning from those transitions.
        intrinsic_rewards, scored_state = self.ensemble.get_info_gain(
            input=model_input,
            state=state,
            update_normalizer=True,
        )

        model_output = ensemble_targets_from_modalities(
            features,
            next_features,
            extrinsic_rewards,
            self.predict_reward,
            self.predict_diff,
            self.output_modalities,
        )
        # The first fit also updates input/output normalizers. Additional
        # epochs reuse those statistics so this rollout is counted once, not
        # once per optimizer epoch.
        new_state, (loss, mse) = self.ensemble.update(
            input=model_input,
            output=model_output,
            state=scored_state,
        )
        losses = [loss]
        mses = [mse]
        for _ in range(1, self.num_update_epochs):
            new_state, (loss, mse) = _update_ensemble_params(
                self.ensemble, model_input, model_output, new_state)
            losses.append(loss)
            mses.append(mse)
        loss = jnp.mean(jnp.stack(losses))
        mse = jnp.mean(jnp.stack(mses))

        normalizers = new_state.ensemble_normalizer_state
        metrics = {
            "ensemble_reward_mean": jnp.mean(intrinsic_rewards),
            "ensemble_reward_std": jnp.std(intrinsic_rewards),
            "expl_reward": jnp.mean(intrinsic_rewards),
            "ens_nll": loss,
            "ens_mse": mse,
            "dyn_model_loss": loss,
            "dyn_model_mse": mse,
            "ens_info_gain_mean": (
                normalizers.info_gain_normalizer_state.mean.mean()),
            "ens_info_gain_std": (
                normalizers.info_gain_normalizer_state.std.mean()),
            "ensemble_step": jnp.asarray(new_state.step),
        }
        return intrinsic_rewards, new_state, metrics


# Descriptive alias for callers that prefer the subsystem name over the
# algorithm-prefixed class name.
EnsembleDisagreementReward = DPPODisagreementReward


__all__ = (
    "DPPODisagreementReward",
    "EnsembleDisagreementReward",
    "MODALITY_ORDER",
    "modality_features",
)
