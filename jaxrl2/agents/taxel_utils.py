"""Shared summed-taxel signals for tactile exploration baselines."""

from typing import Mapping

import jax.numpy as jnp


def summed_taxel_magnitude(tactile: jnp.ndarray) -> jnp.ndarray:
    """Return one non-negative L1 tactile scalar per batch item.

    LIBERO exposes signed, symlog-scaled XYZ force components and the DSRL
    replay buffer appends a frame-stack axis.  Summing absolute values across
    every non-batch axis matches the repository's existing ``touch_sum``
    diagnostic and prevents opposing force components from cancelling.

    Args:
        tactile: Batched tactile observations with shape ``(B, ...)``.

    Returns:
        An array with shape ``(B, 1)``.
    """
    tactile = jnp.asarray(tactile, dtype=jnp.float32)
    if tactile.ndim < 2:
        raise ValueError(
            "Expected batched tactile observations with shape (B, ...); "
            f"got {tactile.shape}.")
    reduction_axes = tuple(range(1, tactile.ndim))
    return jnp.sum(jnp.abs(tactile), axis=reduction_axes, keepdims=False)[:, None]


def summed_taxel_reward(
        batch: Mapping,
        scale: float = 1.0,
) -> jnp.ndarray:
    """Compute transition rewards from the resulting tactile observation.

    The action stored in a replay transition produces ``next_observations``,
    so its contact-signal reward must be taken from that side of the transition.
    The returned shape follows ``batch['rewards']`` for drop-in SAC use.
    """
    try:
        tactile = batch["next_observations"]["tactile"]
    except KeyError as exc:
        raise ValueError(
            "The summed-taxel reward requires tactile data in "
            "batch['next_observations']['tactile'].") from exc

    rewards = jnp.asarray(batch["rewards"])
    taxel_rewards = summed_taxel_magnitude(tactile)[:, 0]
    if rewards.size != taxel_rewards.size:
        raise ValueError(
            "Tactile and reward batch sizes do not match: "
            f"{taxel_rewards.shape} versus {rewards.shape}.")
    return (float(scale) * taxel_rewards).reshape(rewards.shape)
