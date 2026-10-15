from typing import Sequence, Tuple, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from jaxrl.agents.drq.networks import Encoder
from jaxrl.agents.drq.networks import DrQDoubleCritic
from jaxrl.networks.critic_net import DoubleCritic
from jaxrl.networks.policies import MSEPolicy
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class MaxInfoMBDrQv2DoubleCritic(DrQDoubleCritic):
    obs_dim: int = 64

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray, state: Optional[jnp.ndarray] = None) \
            -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        if state is None:
            x = Encoder(self.cnn_features,
                        self.cnn_strides,
                        self.cnn_padding,
                        name='SharedEncoder')(observations)

            window_size = x.shape[-1] // self.obs_dim
            stride_size = x.shape[-1] // self.obs_dim
            obs = jax.lax.reduce_window(
                x,
                init_value=-jnp.inf,  # Initialize with negative infinity for max pooling
                computation=jax.lax.max,  # Use max as the reduction operation
                window_dimensions=(1, window_size),  # Pooling window size
                window_strides=(1, stride_size),  # Stride for the window
                padding='VALID'  # Use valid padding (no padding)
            )

            x = nn.Dense(self.latent_dim)(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)
        else:
            assert state.shape[-1] == self.latent_dim
            x = state
            obs = None

        critic = DoubleCritic(self.hidden_dims)(x, actions)
        expl_critic = DoubleCritic(self.hidden_dims)(x, actions)
        return critic, expl_critic, x, obs


class MaxInfoMBDrQv2Policy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    cnn_features: Sequence[int] = (32, 32, 32, 32)
    cnn_strides: Sequence[int] = (2, 1, 1, 1)
    cnn_padding: str = 'VALID'
    latent_dim: int = 50

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0,
                 state: Optional[jnp.ndarray] = None,
                 ) -> tfd.Distribution:
        if state is None:
            x = Encoder(self.cnn_features,
                        self.cnn_strides,
                        self.cnn_padding,
                        name='SharedEncoder')(observations)

            # We do not update conv layers with policy gradients.
            x = jax.lax.stop_gradient(x)

            x = nn.Dense(self.latent_dim)(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)
        else:
            assert state.shape[-1] == self.latent_dim
            x = state
        return MSEPolicy(self.hidden_dims, self.action_dim)(x, temperature)