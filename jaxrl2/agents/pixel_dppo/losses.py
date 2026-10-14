"""Pure JAX loss helpers for diffusion policy policy optimization.

The helpers in this module operate on already-sampled denoising transitions.
They intentionally do not depend on pi-zero or on any learner/network class so
that the DPPO-specific probability and clipping conventions remain testable in
isolation.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import jax.numpy as jnp


Array = jnp.ndarray


def gaussian_transition_log_prob(
        next_sample: Array,
        mean: Array,
        std: Array,
        min_std: float = 0.1,
) -> Array:
    """Returns elementwise Gaussian log probabilities for a denoising step.

    ``std`` is floored before evaluating the likelihood, matching DPPO's
    ``min_logprob_denoising_std`` stabilization. Inputs follow normal JAX
    broadcasting rules and the result has the broadcast action-chain shape.
    No reduction is performed here.
    """
    next_sample = jnp.asarray(next_sample)
    mean = jnp.asarray(mean, dtype=next_sample.dtype)
    std = jnp.asarray(std, dtype=next_sample.dtype)
    min_std = jnp.asarray(min_std, dtype=next_sample.dtype)

    likelihood_std = jnp.maximum(std, min_std)
    normalized_error = (next_sample - mean) / likelihood_std
    log_two_pi = jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=next_sample.dtype))
    return -0.5 * (
        jnp.square(normalized_error)
        + 2.0 * jnp.log(likelihood_std)
        + log_two_pi
    )


def masked_mean_log_prob(
        per_coordinate_log_probs: Array,
        executed_horizon: Array | int,
        real_action_dim: Array | int,
        coordinate_mask: Optional[Array] = None,
        clamp_min: float = -5.0,
        clamp_max: float = 2.0,
) -> Array:
    """Clamps and averages log probabilities over reward-relevant entries.

    The final two axes must be ``(action_horizon, action_dim)``. Only action
    positions before ``executed_horizon`` and dimensions before
    ``real_action_dim`` contribute. Either bound may be a scalar or broadcast
    to the leading batch shape. ``coordinate_mask`` can additionally remove
    arbitrary entries and must be broadcastable to the full input shape.

    The official DPPO implementation averages, rather than sums, coordinate
    log probabilities before forming the PPO ratio. A sample with no selected
    coordinates returns NaN so an invalid rollout mask cannot pass silently.
    """
    log_probs = jnp.asarray(per_coordinate_log_probs)
    if log_probs.ndim < 2:
        raise ValueError(
            "per_coordinate_log_probs must have action-horizon and action-dim "
            "axes."
        )

    batch_shape = log_probs.shape[:-2]
    horizon_size, action_size = log_probs.shape[-2:]
    executed_horizon = jnp.broadcast_to(
        jnp.asarray(executed_horizon), batch_shape
    )
    real_action_dim = jnp.broadcast_to(
        jnp.asarray(real_action_dim), batch_shape
    )

    horizon_mask = (
        jnp.arange(horizon_size) < executed_horizon[..., None]
    )[..., :, None]
    action_mask = (
        jnp.arange(action_size) < real_action_dim[..., None]
    )[..., None, :]
    mask = jnp.logical_and(horizon_mask, action_mask)
    if coordinate_mask is not None:
        mask = jnp.logical_and(
            mask,
            jnp.broadcast_to(jnp.asarray(coordinate_mask, dtype=bool), log_probs.shape),
        )

    mask = jnp.broadcast_to(mask, log_probs.shape)
    clipped_log_probs = jnp.clip(log_probs, clamp_min, clamp_max)
    selected_sum = jnp.sum(
        jnp.where(mask, clipped_log_probs, jnp.zeros_like(clipped_log_probs)),
        axis=(-2, -1),
    )
    selected_count = jnp.sum(mask, axis=(-2, -1))
    return jnp.where(
        selected_count > 0,
        selected_sum / selected_count.astype(log_probs.dtype),
        jnp.asarray(jnp.nan, dtype=log_probs.dtype),
    )


def exponential_clip_epsilon(
        denoising_indices: Array,
        num_finetune_steps: int,
        clip_epsilon: float,
        clip_epsilon_base: float = 1e-3,
        rate: float = 3.0,
) -> Array:
    """Returns DPPO's denoising-step-dependent PPO clipping threshold.

    Index zero denotes the noisiest selected transition and index
    ``num_finetune_steps - 1`` the cleanest. The threshold grows
    exponentially from ``clip_epsilon_base`` to ``clip_epsilon``. For a
    single fine-tuned step this follows the official implementation and
    returns zero.
    """
    if num_finetune_steps <= 0:
        raise ValueError("num_finetune_steps must be positive.")

    indices = jnp.asarray(denoising_indices)
    dtype = jnp.result_type(indices, jnp.float32)
    indices = indices.astype(dtype)
    denominator_steps = max(num_finetune_steps - 1, 1)
    normalized_step = indices / jnp.asarray(denominator_steps, dtype=dtype)
    if num_finetune_steps == 1:
        return normalized_step

    clip_epsilon = jnp.asarray(clip_epsilon, dtype=dtype)
    clip_epsilon_base = jnp.asarray(clip_epsilon_base, dtype=dtype)
    rate = jnp.asarray(rate, dtype=dtype)
    interpolation = (
        jnp.exp(rate * normalized_step) - 1.0
    ) / (jnp.exp(rate) - 1.0)
    return clip_epsilon_base + (
        clip_epsilon - clip_epsilon_base
    ) * interpolation


def discount_denoising_advantages(
        advantages: Array,
        denoising_indices: Array,
        num_finetune_steps: int,
        gamma_denoising: float,
) -> Array:
    """Downweights advantages for noisier denoising transitions.

    This applies ``gamma_denoising ** (K' - index - 1)``. Thus the cleanest
    selected transition is undiscounted while earlier/noisier transitions
    receive successively smaller weights.
    """
    if num_finetune_steps <= 0:
        raise ValueError("num_finetune_steps must be positive.")

    advantages = jnp.asarray(advantages)
    indices = jnp.asarray(denoising_indices)
    exponent = num_finetune_steps - indices - 1
    discount = jnp.power(
        jnp.asarray(gamma_denoising, dtype=advantages.dtype),
        exponent,
    )
    return advantages * discount


def clipped_ppo_loss(
        new_log_probs: Array,
        old_log_probs: Array,
        advantages: Array,
        clip_epsilon: Array | float,
) -> Tuple[Array, Dict[str, Array]]:
    """Computes DPPO's clipped surrogate loss and official diagnostics.

    ``clip_epsilon`` may vary per sample, as it does under the exponential
    denoising-step schedule. The approximate KL is Schulman's nonnegative k3
    estimator ``(ratio - 1) - log_ratio``.
    """
    new_log_probs = jnp.asarray(new_log_probs)
    old_log_probs = jnp.asarray(old_log_probs, dtype=new_log_probs.dtype)
    advantages = jnp.asarray(advantages, dtype=new_log_probs.dtype)
    clip_epsilon = jnp.asarray(clip_epsilon, dtype=new_log_probs.dtype)

    log_ratio = new_log_probs - old_log_probs
    ratio = jnp.exp(log_ratio)
    policy_loss_unclipped = -advantages * ratio
    policy_loss_clipped = -advantages * jnp.clip(
        ratio,
        1.0 - clip_epsilon,
        1.0 + clip_epsilon,
    )
    loss = jnp.mean(jnp.maximum(
        policy_loss_unclipped,
        policy_loss_clipped,
    ))
    approx_kl = jnp.mean((ratio - 1.0) - log_ratio)
    clip_fraction = jnp.mean(
        (jnp.abs(ratio - 1.0) > clip_epsilon).astype(jnp.float32)
    )
    return loss, {
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
        "ratio": jnp.mean(ratio),
    }


__all__ = (
    "clipped_ppo_loss",
    "discount_denoising_advantages",
    "exponential_clip_epsilon",
    "gaussian_transition_log_prob",
    "masked_mean_log_prob",
)
