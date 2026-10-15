from jaxrl.agents.drq.networks import DrQDoubleCritic
import jax.numpy as jnp
from jaxrl.networks.critic_net import DoubleCritic
from jaxrl.agents.drq.networks import Encoder
import jax
import chex
from jaxtyping import PyTree
from typing import Sequence, Tuple, Optional
from jaxrl.networks.common import default_init
from jaxrl.networks.policies import NormalTanhPolicy, MSEPolicy
import flax.linen as nn
from jaxrl2.utils.tactile.utils import TactileObs
from tensorflow_probability.substrates import jax as tfp
from jaxrl.networks.common import MLP

tfd = tfp.distributions


@chex.dataclass
class TactileEncoding:
    img_embed: PyTree
    tact_embed: Optional[PyTree]
    full_tact: Optional[PyTree]


class TactileEncoder(nn.Module):
    features: Sequence[int] = (32, 32, 32, 32)
    strides: Sequence[int] = (2, 1, 1, 1)
    padding: str = 'VALID'
    tact_features: Sequence[int] = (256, 256)
    mask_touch: bool = False

    @nn.compact
    def __call__(self, observations: TactileObs) -> TactileEncoding:
        assert len(self.features) == len(self.strides)
        x_img = observations.image.astype(jnp.float32) / 255.0
        x_tactile = observations.tactile.astype(jnp.float32)
        if len(x_tactile.shape) in [1, 2]:
            use_cnn = False
        elif len(x_tactile.shape) in [3, 4]:
            use_cnn = True
        else:
            raise NotImplementedError
        if use_cnn:
            if len(x_tactile.shape) == 4:
                # boolean indicator for if touch takes place
                touch = jax.vmap(lambda x: jnp.max(jnp.abs(x)) > 0)(x_tactile)
            else:
                touch = jnp.max(jnp.abs(x_tactile)) > 0

            if not self.mask_touch:
                touch = jnp.ones_like(touch)

            for features, stride in zip(self.features, self.strides):
                x_img = nn.Conv(features,
                                kernel_size=(3, 3),
                                strides=(stride, stride),
                                kernel_init=default_init(),
                                padding=self.padding)(x_img)
                x_img = nn.relu(x_img)

                x_tactile = nn.Conv(features,
                                    kernel_size=(3, 3),
                                    strides=(stride, stride),
                                    kernel_init=default_init(),
                                    padding=self.padding)(x_tactile)
                x_tactile = nn.relu(x_tactile)
            full_tact = x_tactile

            if len(x_img.shape) == 4:
                x_img = x_img.reshape([x_img.shape[0], -1])
            else:
                x_img = x_img.reshape([-1])

            if len(x_tactile.shape) == 4:
                x_tactile = x_tactile.reshape([x_tactile.shape[0], -1])
                touch = touch[..., jnp.newaxis]
            else:
                # replace the output of the encoder with 0s if touch doesn't take place
                x_tactile = x_tactile.reshape([-1])

            # replace the output of the encoder with 0s if touch doesn't take place
            x_tactile = touch * x_tactile

        else:
            if len(x_tactile.shape) == 2:
                touch = jax.vmap(lambda x: jnp.max(jnp.abs(x)) > 0)(x_tactile)
                touch = touch[..., jnp.newaxis]
            else:
                touch = jnp.max(jnp.abs(x_tactile)) > 0

            if not self.mask_touch:
                touch = jnp.ones_like(touch)

            for features, stride in zip(self.features, self.strides):
                x_img = nn.Conv(features,
                                kernel_size=(3, 3),
                                strides=(stride, stride),
                                kernel_init=default_init(),
                                padding=self.padding)(x_img)
                x_img = nn.relu(x_img)
            if len(x_img.shape) == 4:
                x_img = x_img.reshape([x_img.shape[0], -1])
            else:
                x_img = x_img.reshape([-1])

            x_tactile = MLP(hidden_dims=self.tact_features)(x_tactile)
            full_tact = x_tactile
            # replace the output of the encoder with 0s if touch doesn't take place
            x_tactile = touch * x_tactile

        return TactileEncoding(
            img_embed=x_img,
            tact_embed=x_tactile,
            full_tact=full_tact,
        )


class TactileDecoder(nn.Module):
    features: Sequence[int] = (32, 32, 32, 32)
    strides: Sequence[int] = (2, 1, 1, 1)
    padding: str = 'VALID'
    tact_features: Sequence[int] = (256, 256)
    tact_input_shape: Tuple = (16, 16, 18)

    @nn.compact
    def __call__(self, tactile_encoding: TactileEncoding) -> jnp.ndarray:
        assert len(self.features) == len(self.strides)
        if len(self.tact_input_shape) in [1, 2]:
            use_cnn = False
        elif len(self.tact_input_shape) in [3, 4]:
            use_cnn = True
        else:
            raise NotImplementedError
        decoding = tactile_encoding.full_tact
        if use_cnn:
            for features, stride in zip(reversed(self.features), reversed(self.strides)):
                decoding = nn.ConvTranspose(features,
                                            kernel_size=(3, 3),
                                            strides=(stride, stride),
                                            kernel_init=default_init(),
                                            padding=self.padding)(decoding)
                decoding = nn.relu(decoding)

            desired_out = self.tact_input_shape[-1]
            kernel_size = self.tact_input_shape[-2] + 1 - decoding.shape[-2]
            assert kernel_size > 0
            decoding = nn.ConvTranspose(desired_out,
                                        kernel_size=(kernel_size, kernel_size),
                                        strides=(1, 1),
                                        kernel_init=default_init(),
                                        padding=self.padding)(decoding)
        else:
            decoding = MLP(hidden_dims=self.tact_features + (self.tact_input_shape[-1],))(decoding)

        return decoding


class TactileDrQDoubleCritic(DrQDoubleCritic):
    tact_features: Sequence[int] = (256, 256)
    mask_touch: bool = False

    @nn.compact
    def __call__(self, observations: TactileObs,
                 actions: jnp.ndarray) -> jnp.ndarray:  # Tuple[jnp.ndarray, jnp.ndarray, TactileEncoding]:
        # if we have tactile observation, use visual-tactile encoder, else use visual encoder
        if observations.tactile is not None:
            embedding = TactileEncoder(
                self.cnn_features, self.cnn_strides, self.cnn_padding,
                self.tact_features,
                mask_touch=self.mask_touch,
                name='SharedEncoder')(observations)

            x = jnp.concatenate([embedding.img_embed, embedding.tact_embed], axis=-1)
        else:
            x = Encoder(
                self.cnn_features, self.cnn_strides, self.cnn_padding, name='SharedEncoder')(observations.image)

        x = nn.Dense(self.latent_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)

        return DoubleCritic(self.hidden_dims)(x, actions)


class TactileDrQPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    cnn_features: Sequence[int] = (32, 32, 32, 32)
    cnn_strides: Sequence[int] = (2, 1, 1, 1)
    cnn_padding: str = 'VALID'
    latent_dim: int = 50
    tact_features: Sequence[int] = (256, 256)
    mask_touch: bool = False

    @nn.compact
    def __call__(self,
                 observations: TactileObs,
                 temperature: float = 1.0) -> tfd.Distribution:
        if observations.tactile is not None:
            embedding = TactileEncoder(
                self.cnn_features, self.cnn_strides, self.cnn_padding,
                self.tact_features,
                mask_touch=self.mask_touch,
                name='SharedEncoder')(observations)

            x = jnp.concatenate([embedding.img_embed, embedding.tact_embed], axis=-1)
        else:
            x = Encoder(
                self.cnn_features, self.cnn_strides, self.cnn_padding, name='SharedEncoder')(observations.image)
        # We do not update conv layers with policy gradients.
        x = jax.lax.stop_gradient(x)

        x = nn.Dense(self.latent_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)

        return NormalTanhPolicy(self.hidden_dims, self.action_dim)(x, temperature)


class TactileDrQv2Policy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    cnn_features: Sequence[int] = (32, 32, 32, 32)
    cnn_strides: Sequence[int] = (2, 1, 1, 1)
    cnn_padding: str = 'VALID'
    latent_dim: int = 50
    tact_features: Sequence[int] = (256, 256)
    mask_touch: bool = False

    @nn.compact
    def __call__(self,
                 observations: TactileObs,
                 temperature: float = 1.0) -> tfd.Distribution:
        if observations.tactile is not None:
            embedding = TactileEncoder(
                self.cnn_features, self.cnn_strides, self.cnn_padding,
                self.tact_features,
                mask_touch=self.mask_touch,
                name='SharedEncoder')(observations)

            x = jnp.concatenate([embedding.img_embed, embedding.tact_embed], axis=-1)
        else:
            x = Encoder(
                self.cnn_features, self.cnn_strides, self.cnn_padding, name='SharedEncoder')(observations.image)
        # We do not update conv layers with policy gradients.
        x = jax.lax.stop_gradient(x)

        x = nn.Dense(self.latent_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)

        return MSEPolicy(self.hidden_dims, self.action_dim)(x, temperature)


class TactileMaxInfoDrQDoubleCritic(DrQDoubleCritic):
    tact_features: Sequence[int] = (256, 256)
    obs_dim: int = 64
    mask_touch: bool = False
    add_decoder: bool = False

    @nn.compact
    def __call__(self, observations: TactileObs,
                 actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, TactileEncoding]:
        if observations.tactile is not None:
            embedding = TactileEncoder(
                self.cnn_features, self.cnn_strides, self.cnn_padding,
                self.tact_features,
                mask_touch=self.mask_touch,
                name='SharedEncoder')(observations)
            if self.add_decoder:
                decoding = TactileDecoder(
                    self.cnn_features, self.cnn_strides, self.cnn_padding,
                    self.tact_features,
                    tact_input_shape=observations.tactile.shape[1:],
                    name='SharedDecoder',
                )(embedding)
                embedding = embedding.replace(
                    full_tact=decoding,
                )
            x = jnp.concatenate([embedding.img_embed, embedding.tact_embed], axis=-1)
        else:
            x = Encoder(
                self.cnn_features, self.cnn_strides, self.cnn_padding, name='SharedEncoder')(observations.image)
            embedding = TactileEncoding(
                img_embed=x,
                tact_embed=None,
                full_tact=None,
            )

        x = nn.Dense(self.latent_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)

        window_size = embedding.img_embed.shape[-1] // self.obs_dim
        stride_size = embedding.img_embed.shape[-1] // self.obs_dim

        img_embed = jax.lax.reduce_window(
            embedding.img_embed,
            init_value=-jnp.inf,  # Initialize with negative infinity for max pooling
            computation=jax.lax.max,  # Use max as the reduction operation
            window_dimensions=(1, window_size),  # Pooling window size
            window_strides=(1, stride_size),  # Stride for the window
            padding='VALID'  # Use valid padding (no padding)
        )

        if embedding.tact_embed is not None:
            window_size = embedding.tact_embed.shape[-1] // self.obs_dim
            stride_size = embedding.tact_embed.shape[-1] // self.obs_dim

            tact_embed = jax.lax.reduce_window(
                embedding.tact_embed,
                init_value=-jnp.inf,  # Initialize with negative infinity for max pooling
                computation=jax.lax.max,  # Use max as the reduction operation
                window_dimensions=(1, window_size),  # Pooling window size
                window_strides=(1, stride_size),  # Stride for the window
                padding='VALID'  # Use valid padding (no padding)
            )
        else:
            tact_embed = None

        embedding = embedding.replace(
            img_embed=img_embed,
            tact_embed=tact_embed,
        )

        return DoubleCritic(self.hidden_dims)(x, actions), x, embedding


class TactileMaxInfoDrQv2DoubleCritic(DrQDoubleCritic):
    tact_features: Sequence[int] = (256, 256)
    obs_dim: int = 64
    mask_touch: bool = False
    add_decoder: bool = False

    @nn.compact
    def __call__(self, observations: TactileObs,
                 actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, TactileEncoding]:
        if observations.tactile is not None:
            embedding = TactileEncoder(
                self.cnn_features, self.cnn_strides, self.cnn_padding,
                self.tact_features,
                mask_touch=self.mask_touch,
                name='SharedEncoder')(observations)
            if self.add_decoder:
                decoding = TactileDecoder(
                    self.cnn_features, self.cnn_strides, self.cnn_padding,
                    self.tact_features,
                    tact_input_shape=observations.tactile.shape[1:],
                    name='SharedDecoder',
                )(embedding)
                embedding = embedding.replace(
                    full_tact=decoding,
                )

            x = jnp.concatenate([embedding.img_embed, embedding.tact_embed], axis=-1)
        else:
            x = Encoder(
                self.cnn_features, self.cnn_strides, self.cnn_padding, name='SharedEncoder')(observations.image)
            embedding = TactileEncoding(
                img_embed=x,
                tact_embed=None,
                full_tact=None,
            )

        x = nn.Dense(self.latent_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)

        window_size = embedding.img_embed.shape[-1] // self.obs_dim
        stride_size = embedding.img_embed.shape[-1] // self.obs_dim

        img_embed = jax.lax.reduce_window(
            embedding.img_embed,
            init_value=-jnp.inf,  # Initialize with negative infinity for max pooling
            computation=jax.lax.max,  # Use max as the reduction operation
            window_dimensions=(1, window_size),  # Pooling window size
            window_strides=(1, stride_size),  # Stride for the window
            padding='VALID'  # Use valid padding (no padding)
        )

        if embedding.tact_embed is not None:
            window_size = embedding.tact_embed.shape[-1] // self.obs_dim
            stride_size = embedding.tact_embed.shape[-1] // self.obs_dim

            tact_embed = jax.lax.reduce_window(
                embedding.tact_embed,
                init_value=-jnp.inf,  # Initialize with negative infinity for max pooling
                computation=jax.lax.max,  # Use max as the reduction operation
                window_dimensions=(1, window_size),  # Pooling window size
                window_strides=(1, stride_size),  # Stride for the window
                padding='VALID'  # Use valid padding (no padding)
            )
        else:
            tact_embed = None

        embedding = embedding.replace(
            img_embed=img_embed,
            tact_embed=tact_embed,
        )

        critic = DoubleCritic(self.hidden_dims)(x, actions)
        expl_critic = DoubleCritic(self.hidden_dims)(x, actions)
        return critic, expl_critic, x, embedding
