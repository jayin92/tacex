"""Network and distribution helpers for pixel-based PPO.

The PPO policy keeps the Gaussian in pre-tanh space and applies the bounded
action transform explicitly.  Besides being numerically stable near the action
bounds, this avoids relying on Distrax's inferred ``Lambda`` inverse, which is
incompatible with the JAX/Distrax versions pinned by this repository.
"""

from typing import Optional, Sequence, Tuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp

from jaxrl2.networks import MLP
from jaxrl2.networks.constants import default_init


class StateIndependentNormalPolicy(nn.Module):
    """Diagonal Gaussian policy with a learned, observation-independent std."""

    hidden_dims: Sequence[int]
    action_dim: int
    dropout_rate: Optional[float] = None
    init_log_std: float = 0.0
    log_std_min: float = -20.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, observations, training: bool = False):
        outputs = MLP(
            self.hidden_dims,
            activate_final=True,
            dropout_rate=self.dropout_rate,
        )(observations, training=training)
        means = nn.Dense(
            self.action_dim,
            kernel_init=default_init(1e-2),
            name="means",
        )(outputs)
        log_stds = self.param(
            "log_stds",
            nn.initializers.constant(self.init_log_std),
            (self.action_dim,),
        )
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        scale_diag = jnp.broadcast_to(jnp.exp(log_stds), means.shape)
        return distrax.MultivariateNormalDiag(
            loc=means,
            scale_diag=scale_diag,
        )


def tanh_log_det_jacobian(
        pre_tanh_action: jnp.ndarray,
        action_magnitude: float = 1.0,
) -> jnp.ndarray:
    """Returns ``log |d(magnitude * tanh(x)) / dx|`` per batch item.

    The softplus identity remains finite for large positive or negative inputs,
    unlike computing ``log(1 - tanh(x) ** 2)`` directly.
    """

    pre_tanh_action = jnp.asarray(pre_tanh_action)
    log_tanh_derivative = 2.0 * (
        jnp.log(2.0)
        - pre_tanh_action
        - jax.nn.softplus(-2.0 * pre_tanh_action)
    )
    log_scale = jnp.log(jnp.asarray(action_magnitude, dtype=pre_tanh_action.dtype))
    return jnp.sum(log_tanh_derivative + log_scale, axis=-1)


def squash_action(
        pre_tanh_action: jnp.ndarray,
        action_magnitude: float = 1.0,
) -> jnp.ndarray:
    """Maps an unconstrained Gaussian action into the DSRL action bounds."""

    return jnp.asarray(action_magnitude) * jnp.tanh(pre_tanh_action)


def squashed_log_prob(
        distribution: distrax.MultivariateNormalDiag,
        pre_tanh_action: jnp.ndarray,
        action_magnitude: float = 1.0,
) -> jnp.ndarray:
    """Evaluates the transformed density without inverting a saturated tanh."""

    return (
        distribution.log_prob(pre_tanh_action)
        - tanh_log_det_jacobian(pre_tanh_action, action_magnitude)
    )


def sample_squashed_action_and_log_prob(
        distribution: distrax.MultivariateNormalDiag,
        seed: jax.Array,
        action_magnitude: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Samples a bounded action and returns its pre-tanh value and log density."""

    pre_tanh_action = distribution.sample(seed=seed)
    action = squash_action(pre_tanh_action, action_magnitude)
    log_prob = squashed_log_prob(
        distribution,
        pre_tanh_action,
        action_magnitude,
    )
    return action, pre_tanh_action, log_prob
