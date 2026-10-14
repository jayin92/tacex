"""Trainable stochastic pi-zero sampler for the isolated DPPO baseline.

Pi-zero uses a deterministic rectified-flow Euler sampler after drawing its
initial noise.  DPPO, however, requires a stochastic Markov transition at each
denoising step so that PPO can evaluate a behavior/current-policy likelihood
ratio.  This module implements the smallest explicit adaptation of the paper:

``x_next ~ Normal(x + dt * v_theta(o, x, t), sigma**2 I)``.

The regular OpenPI ``Policy`` object is deliberately not modified.  It freezes
model state inside its jitted inference closure, which is ideal for the other
experiments but cannot reflect online DPPO updates.
"""

from __future__ import annotations

import copy
import functools
import pathlib
from collections.abc import Mapping, Sequence
from typing import Any

import einops
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrl2.agents.pixel_dppo.losses import (
    clipped_ppo_loss,
    discount_denoising_advantages,
    exponential_clip_epsilon,
    gaussian_transition_log_prob,
    masked_mean_log_prob,
)
from openpi import transforms as openpi_transforms
from openpi.models import model as openpi_model
from openpi.models.pi0 import make_attn_mask
from openpi.shared import nnx_utils
from openpi.shared import download
from openpi.training import checkpoints as openpi_checkpoints


def _trainable_filter(scope: str):
    projections = nnx_utils.PathRegex(
        r"(state_proj|action_in_proj|action_out_proj|"
        r"action_time_mlp_in|action_time_mlp_out)/.*")
    output_head = nnx_utils.PathRegex(r"action_out_proj/.*")
    action_expert = nnx_utils.PathRegex(r"PaliGemma/llm/.*_1/.*")

    if scope == "action_head":
        path_filter = output_head
    elif scope == "action_projections":
        path_filter = projections
    elif scope == "action_expert":
        path_filter = nnx.Any(projections, action_expert)
    elif scope == "all":
        return nnx.Param
    else:
        raise ValueError(
            "dppo_trainable_scope must be one of action_head, "
            f"action_projections, action_expert, or all; got {scope!r}.")
    return nnx.All(nnx.Param, path_filter)


def _prefix_cache(model, observation):
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm(
        [prefix_tokens, None],
        mask=prefix_attn_mask,
        positions=positions,
    )
    return prefix_tokens, prefix_mask, kv_cache


def _flow_transition_mean(
        model,
        observation,
        x_t,
        time,
        prefix_tokens,
        prefix_mask,
        kv_cache,
        num_denoising_steps: int,
):
    """Computes one pi-zero Euler transition mean for a batch."""
    batch_size = observation.state.shape[0]
    suffix_tokens, suffix_mask, suffix_ar_mask = model.embed_suffix(
        observation, x_t, time)
    suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    prefix_attn_mask = einops.repeat(
        prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
    full_attn_mask = jnp.concatenate(
        [prefix_attn_mask, suffix_attn_mask], axis=-1)
    positions = (
        jnp.sum(prefix_mask, axis=-1)[:, None]
        + jnp.cumsum(suffix_mask, axis=-1)
        - 1
    )
    (prefix_out, suffix_out), _ = model.PaliGemma.llm(
        [None, suffix_tokens],
        mask=full_attn_mask,
        positions=positions,
        kv_cache=kv_cache,
    )
    if prefix_out is not None:
        raise AssertionError("Cached pi-zero suffix pass returned a prefix.")
    velocity = model.action_out_proj(
        suffix_out[:, -model.action_horizon:])
    dt = jnp.asarray(-1.0 / num_denoising_steps, dtype=x_t.dtype)
    return x_t + dt * velocity.astype(x_t.dtype)


def _sample_chain_impl(
        model_def,
        trainable_state,
        frozen_state,
        base_trainable_state,
        rng,
        observation,
        *,
        num_denoising_steps: int,
        finetune_denoising_steps: int,
        sampling_std: float,
        logprob_std: float,
        deterministic: bool,
        share_prefix: bool,
):
    # The pretrained, non-trainable leaves are shared by both models.  Keeping
    # only a snapshot of the small trainable partition avoids materializing a
    # second copy of the multi-billion-parameter pi-zero checkpoint.
    model = nnx.merge(
        model_def, nnx.State.merge(frozen_state, trainable_state))
    base_model = nnx.merge(
        model_def, nnx.State.merge(frozen_state, base_trainable_state))
    observation = openpi_model.preprocess_observation(
        None, observation, train=False)

    current_prefix = _prefix_cache(model, observation)
    base_prefix = (
        current_prefix
        if share_prefix
        else _prefix_cache(base_model, observation)
    )
    batch_size = observation.state.shape[0]
    rng, initial_key = jax.random.split(rng)
    x_t = jax.random.normal(
        initial_key,
        (batch_size, model.action_horizon, model.action_dim),
        dtype=jnp.float32,
    )

    selected_start = num_denoising_steps - finetune_denoising_steps
    selected_chain = []
    old_log_probs = []
    for step_index in range(num_denoising_steps):
        use_finetuned = step_index >= selected_start
        step_model = model if use_finetuned else base_model
        prefix_tokens, prefix_mask, kv_cache = (
            current_prefix if use_finetuned else base_prefix)
        time = jnp.full(
            (batch_size,),
            1.0 - step_index / num_denoising_steps,
            dtype=x_t.dtype,
        )
        mean = _flow_transition_mean(
            step_model,
            observation,
            x_t,
            time,
            prefix_tokens,
            prefix_mask,
            kv_cache,
            num_denoising_steps,
        )
        if use_finetuned and not selected_chain:
            selected_chain.append(x_t)

        if deterministic:
            x_next = mean
        else:
            rng, noise_key = jax.random.split(rng)
            transition_noise = jnp.clip(
                jax.random.normal(noise_key, x_t.shape, dtype=x_t.dtype),
                -3.0,
                3.0,
            )
            x_next = mean + jnp.asarray(
                sampling_std, dtype=x_t.dtype) * transition_noise

        if use_finetuned:
            selected_chain.append(x_next)
            old_log_probs.append(gaussian_transition_log_prob(
                x_next,
                mean,
                jnp.asarray(sampling_std, dtype=x_t.dtype),
                min_std=logprob_std,
            ))
        x_t = x_next

    return {
        "normalized_actions": x_t,
        "chains": jnp.stack(selected_chain, axis=1),
        "old_log_probs": jnp.stack(old_log_probs, axis=1),
    }


def _update_actor_impl(
        model_def,
        trainable_filter,
        tx,
        trainable_state,
        frozen_state,
        opt_state,
        observation,
        chain_prev,
        chain_next,
        denoising_indices,
        old_per_coordinate_log_probs,
        advantages,
        executed_horizon,
        real_action_dim,
        coordinate_mask,
        *,
        num_denoising_steps: int,
        finetune_denoising_steps: int,
        sampling_std: float,
        logprob_std: float,
        gamma_denoising: float,
        clip_epsilon: float,
        clip_epsilon_base: float,
        clip_epsilon_rate: float,
        entropy_coefficient: float,
):
    model = nnx.merge(
        model_def, nnx.State.merge(frozen_state, trainable_state))
    observation = openpi_model.preprocess_observation(
        None, observation, train=False)

    def actor_loss_fn(diff_model):
        prefix_tokens, prefix_mask, kv_cache = _prefix_cache(
            diff_model, observation)
        global_indices = (
            num_denoising_steps
            - finetune_denoising_steps
            + denoising_indices
        )
        time = 1.0 - global_indices.astype(jnp.float32) / float(
            num_denoising_steps)
        means = _flow_transition_mean(
            diff_model,
            observation,
            chain_prev,
            time,
            prefix_tokens,
            prefix_mask,
            kv_cache,
            num_denoising_steps,
        )
        new_coordinate_log_probs = gaussian_transition_log_prob(
            chain_next,
            means,
            jnp.asarray(sampling_std, dtype=chain_next.dtype),
            min_std=logprob_std,
        )
        new_log_probs = masked_mean_log_prob(
            new_coordinate_log_probs,
            executed_horizon,
            real_action_dim,
            coordinate_mask=coordinate_mask,
        )
        old_log_probs = masked_mean_log_prob(
            old_per_coordinate_log_probs,
            executed_horizon,
            real_action_dim,
            coordinate_mask=coordinate_mask,
        )

        denoising_advantages = discount_denoising_advantages(
            advantages,
            denoising_indices,
            finetune_denoising_steps,
            gamma_denoising,
        )
        step_clip = exponential_clip_epsilon(
            denoising_indices,
            finetune_denoising_steps,
            clip_epsilon,
            clip_epsilon_base,
            clip_epsilon_rate,
        )
        policy_loss, policy_info = clipped_ppo_loss(
            new_log_probs,
            old_log_probs,
            denoising_advantages,
            step_clip,
        )
        likelihood_std = jnp.asarray(logprob_std, dtype=jnp.float32)
        entropy = 0.5 * jnp.log(
            2.0 * jnp.pi * jnp.e * jnp.square(likelihood_std))
        actor_loss = policy_loss - entropy_coefficient * entropy
        return actor_loss, {
            **policy_info,
            "actor_loss": actor_loss,
            "policy_loss": policy_loss,
            "entropy": entropy,
            "new_log_prob": jnp.mean(new_log_probs),
            "old_log_prob": jnp.mean(old_log_probs),
            "advantage": jnp.mean(denoising_advantages),
            "clip_epsilon": jnp.mean(step_clip),
        }

    diff_state = nnx.DiffState(0, trainable_filter)
    (loss, info), grads = nnx.value_and_grad(
        actor_loss_fn,
        argnums=diff_state,
        has_aux=True,
    )(model)
    updates, new_opt_state = tx.update(grads, opt_state, trainable_state)
    new_trainable_params = optax.apply_updates(trainable_state, updates)
    nnx.update(model, new_trainable_params)
    new_trainable_state = nnx.state(model).filter(trainable_filter)
    return new_trainable_state, new_opt_state, {
        **info,
        "actor_grad_norm": optax.global_norm(grads),
        "loss": loss,
    }


class Pi0DPPOPolicy:
    """Owns mutable pi-zero params and DPPO behavior-policy state."""

    def __init__(
            self,
            model,
            *,
            input_transforms: Sequence[Any],
            output_transforms: Sequence[Any],
            learning_rate: float,
            trainable_scope: str,
            num_denoising_steps: int,
            finetune_denoising_steps: int,
            min_sampling_std: float,
            min_logprob_std: float,
            gamma_denoising: float,
            clip_epsilon: float,
            max_grad_norm: float,
            target_kl: float,
            normalize_advantages: bool,
            entropy_coefficient: float = 0.0,
            clip_epsilon_base: float = 1e-3,
            clip_epsilon_rate: float = 3.0,
            seed: int = 0,
    ):
        if num_denoising_steps <= 0:
            raise ValueError("dppo_num_denoising_steps must be positive.")
        if not 0 < finetune_denoising_steps <= num_denoising_steps:
            raise ValueError(
                "dppo_finetune_denoising_steps must be in [1, "
                "dppo_num_denoising_steps].")
        if min_sampling_std <= 0 or min_logprob_std <= 0:
            raise ValueError("DPPO denoising standard deviations must be positive.")
        if not 0 < gamma_denoising <= 1:
            raise ValueError("dppo_gamma_denoising must be in (0, 1].")

        model_def, state = nnx.split(model)
        train_filter = _trainable_filter(trainable_scope)
        if not state.filter(train_filter).flat_state():
            raise ValueError(
                f"No pi-zero parameters matched trainable scope {trainable_scope!r}.")
        trainable_state, frozen_state = state.split(train_filter, ...)
        base_trainable_state = copy.deepcopy(trainable_state)
        # Keep optimizer-controlled leaves in fp32 even though inference-only
        # pi-zero checkpoints are normally restored as bfloat16.
        trainable_state = nnx_utils.state_map(
            trainable_state,
            nnx.Param,
            lambda variable: variable.replace(
                variable.value.astype(jnp.float32)),
        )

        tx = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adamw(learning_rate, weight_decay=0.0),
        )
        self._model_def = model_def
        self._state = trainable_state
        self._frozen_state = frozen_state
        self._base_trainable_state = base_trainable_state
        self._trainable_filter = train_filter
        self._tx = tx
        self._opt_state = tx.init(trainable_state)
        self._input_transform = openpi_transforms.compose(input_transforms)
        self._output_transform = openpi_transforms.compose(output_transforms)
        self._rng = jax.random.PRNGKey(seed)
        self._eval_rng = jax.random.PRNGKey(seed + 1_000_003)
        self._num_updates = 0

        self.action_dim = int(model.action_dim)
        self.action_horizon = int(model.action_horizon)
        self.num_denoising_steps = int(num_denoising_steps)
        self.finetune_denoising_steps = int(finetune_denoising_steps)
        self.min_sampling_std = float(min_sampling_std)
        self.min_logprob_std = float(min_logprob_std)
        self.gamma_denoising = float(gamma_denoising)
        self.clip_epsilon = float(clip_epsilon)
        self.target_kl = float(target_kl)
        self.normalize_advantages = bool(normalize_advantages)
        self.entropy_coefficient = float(entropy_coefficient)
        self.clip_epsilon_base = float(clip_epsilon_base)
        self.clip_epsilon_rate = float(clip_epsilon_rate)
        self.trainable_scope = trainable_scope

        def sample_fn(
                current_trainable_state,
                frozen_state,
                behavior_trainable_state,
                key,
                observation,
                *,
                deterministic,
        ):
            return _sample_chain_impl(
                model_def,
                current_trainable_state,
                frozen_state,
                behavior_trainable_state,
                key,
                observation,
                num_denoising_steps=self.num_denoising_steps,
                finetune_denoising_steps=self.finetune_denoising_steps,
                sampling_std=self.min_sampling_std,
                logprob_std=self.min_logprob_std,
                deterministic=deterministic,
                # Every supported partial scope affects only the action
                # suffix; image/language prefix weights remain frozen.
                share_prefix=trainable_scope != "all",
            )

        self._sample_jit = jax.jit(
            sample_fn, static_argnames=("deterministic",))

        def update_fn(
                current_trainable_state,
                frozen_state,
                opt_state,
                observation,
                chain_prev,
                chain_next,
                denoising_indices,
                old_log_probs,
                advantages,
                executed_horizon,
                real_action_dim,
                coordinate_mask,
        ):
            return _update_actor_impl(
                model_def,
                train_filter,
                tx,
                current_trainable_state,
                frozen_state,
                opt_state,
                observation,
                chain_prev,
                chain_next,
                denoising_indices,
                old_log_probs,
                advantages,
                executed_horizon,
                real_action_dim,
                coordinate_mask,
                num_denoising_steps=self.num_denoising_steps,
                finetune_denoising_steps=self.finetune_denoising_steps,
                sampling_std=self.min_sampling_std,
                logprob_std=self.min_logprob_std,
                gamma_denoising=self.gamma_denoising,
                clip_epsilon=self.clip_epsilon,
                clip_epsilon_base=self.clip_epsilon_base,
                clip_epsilon_rate=self.clip_epsilon_rate,
                entropy_coefficient=self.entropy_coefficient,
            )

        self._update_jit = jax.jit(update_fn)

    def prepare_observation(self, observation: Mapping[str, Any]):
        inputs = jax.tree.map(lambda x: x, observation)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x), inputs)
        if inputs["state"].ndim == 1:
            inputs = jax.tree.map(lambda x: x[jnp.newaxis, ...], inputs)
        elif inputs["state"].ndim != 2:
            raise ValueError(
                "Expected transformed pi-zero state to be rank 1 or 2, got "
                f"shape {inputs['state'].shape}.")
        return openpi_model.Observation.from_dict(inputs)

    def _decode_actions(self, observation, normalized_actions):
        outputs = {
            "state": observation.state,
            "actions": normalized_actions,
        }
        if normalized_actions.shape[0] == 1:
            outputs = jax.tree.map(
                lambda x: np.asarray(jax.device_get(x[0])), outputs)
        else:
            outputs = jax.tree.map(
                lambda x: np.asarray(jax.device_get(x)), outputs)
        return self._output_transform(outputs)["actions"]

    def sample(self, observation, *, deterministic: bool = False):
        if deterministic:
            self._eval_rng, key = jax.random.split(self._eval_rng)
        else:
            self._rng, key = jax.random.split(self._rng)
        result = self._sample_jit(
            self._state,
            self._frozen_state,
            self._base_trainable_state,
            key,
            observation,
            deterministic=deterministic,
        )
        result = jax.tree.map(
            lambda x: np.asarray(jax.device_get(x)), result)
        result["actions"] = np.asarray(self._decode_actions(
            observation, result["normalized_actions"]))
        return result

    def update(
            self,
            observation,
            chain_prev,
            chain_next,
            denoising_indices,
            old_log_probs,
            advantages,
            executed_horizon,
            real_action_dim,
            coordinate_mask,
    ) -> dict[str, float]:
        self._state, self._opt_state, info = self._update_jit(
            self._state,
            self._frozen_state,
            self._opt_state,
            observation,
            chain_prev,
            chain_next,
            denoising_indices,
            old_log_probs,
            advantages,
            executed_horizon,
            real_action_dim,
            coordinate_mask,
        )
        self._num_updates += 1
        return {
            key: float(np.asarray(jax.device_get(value)))
            for key, value in info.items()
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "state": self._state,
            "opt_state": self._opt_state,
            "rng": self._rng,
            "eval_rng": self._eval_rng,
            "num_updates": self._num_updates,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._state = state["state"]
        self._opt_state = state["opt_state"]
        self._rng = state["rng"]
        self._eval_rng = state["eval_rng"]
        self._num_updates = int(state["num_updates"])


def create_pi0_dppo_policy(
        train_config,
        checkpoint_dir: pathlib.Path | str,
        **policy_kwargs,
) -> Pi0DPPOPolicy:
    """Loads pi-zero with the same transforms as regular OpenPI inference."""
    checkpoint_dir = pathlib.Path(download.maybe_download(str(checkpoint_dir)))
    model = train_config.model.load(openpi_model.restore_params(
        checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(
        train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None:
        raise ValueError("pi-zero data config must provide an asset_id.")
    norm_stats = openpi_checkpoints.load_norm_stats(
        checkpoint_dir / "assets", data_config.asset_id)

    return Pi0DPPOPolicy(
        model,
        input_transforms=[
            openpi_transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            openpi_transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            openpi_transforms.Unnormalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        **policy_kwargs,
    )


__all__ = ("Pi0DPPOPolicy", "create_pi0_dppo_policy")
