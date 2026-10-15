from jaxrl.agents.drq.networks import DrQDoubleCritic
from typing import Tuple

import flax.linen as nn
import jax.numpy as jnp
from jaxrl.networks.critic_net import DoubleCritic

from jaxrl.agents.drq.networks import Encoder
import jax


class MaxInfoDrQDoubleCritic(DrQDoubleCritic):
    obs_dim: int = 64

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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

        return DoubleCritic(self.hidden_dims)(x, actions), x, obs
