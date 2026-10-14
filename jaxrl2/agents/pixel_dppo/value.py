"""Observation-value network used by the isolated pi-zero DPPO baseline."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from jaxrl2.agents.pixel_maxinfosac.networks import PixelTactileMultiplexer
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
        return ResNetV2Encoder(
            stage_sizes=(1, 1, 1, 1), norm=encoder_norm)
    if encoder_type == "resnet_18_v2":
        return ResNetV2Encoder(
            stage_sizes=(2, 2, 2, 2), norm=encoder_norm)
    if encoder_type == "resnet_34_v2":
        return ResNetV2Encoder(
            stage_sizes=(3, 4, 6, 3), norm=encoder_norm)
    raise ValueError(f"Unknown encoder type: {encoder_type}")


def _variables(state: TrainState, params):
    variables = {"params": params}
    if state.batch_stats is not None:
        variables["batch_stats"] = state.batch_stats
    return variables


def _apply_value(state: TrainState, params, observations):
    return state.apply_fn(
        _variables(state, params),
        observations,
        training=False,
        mutable=False,
    )


@jax.jit
def _value_jit(state: TrainState, observations):
    return _apply_value(state, state.params, observations)


@jax.jit
def _update_value_jit(
        state: TrainState, observations, returns, loss_coefficient):
    returns = jnp.asarray(returns).reshape(-1)

    def loss_fn(params):
        values = _apply_value(state, params, observations).reshape(-1)
        unscaled_loss = 0.5 * jnp.mean(jnp.square(values - returns))
        loss = jnp.asarray(loss_coefficient) * unscaled_loss
        return loss, {
            "value_loss": unscaled_loss,
            "value": jnp.mean(values),
            "return": jnp.mean(returns),
        }

    (loss, info), grads = jax.value_and_grad(
        loss_fn, has_aux=True)(state.params)
    grad_norm = optax.global_norm(grads)
    new_state = state.apply_gradients(grads=grads)
    return new_state, {**info, "value_grad_norm": grad_norm, "loss": loss}


class DPPOValueLearner:
    """A scalar ``V(o)`` kept independent from the trainable pi-zero actor."""

    def __init__(
            self,
            seed: int,
            observations: Mapping[str, np.ndarray],
            *,
            learning_rate: float = 1e-3,
            max_grad_norm: float = 0.5,
            value_loss_coefficient: float = 0.5,
            hidden_dims: Sequence[int] = (128, 128, 128),
            cnn_features: Sequence[int] = (32, 32, 32, 32),
            cnn_strides: Sequence[int] = (2, 1, 1, 1),
            cnn_padding: str = "VALID",
            latent_dim: int = 50,
            encoder_type: str = "small",
            encoder_norm: str = "group",
            use_spatial_softmax: bool = True,
            softmax_temperature: float = -1.0,
            use_bottleneck: bool = True,
            obs_dim: int = 64,
            tactile_hidden_dims: Sequence[int] = (256, 256),
            mask_touch: bool = False,
    ):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive.")
        if value_loss_coefficient < 0:
            raise ValueError("value_loss_coefficient must be nonnegative.")

        encoder = _make_encoder(
            encoder_type,
            tuple(cnn_features),
            tuple(cnn_strides),
            cnn_padding,
            encoder_norm,
            use_spatial_softmax,
            softmax_temperature,
        )
        value_head = StateValue(tuple(hidden_dims))
        if "tactile" in observations:
            value_def = PixelTactileMultiplexer(
                encoder=encoder,
                network=value_head,
                latent_dim=latent_dim,
                use_bottleneck=use_bottleneck,
                obs_dim=obs_dim,
                tactile_hidden_dims=tuple(tactile_hidden_dims),
                tactile_cnn_features=tuple(cnn_features),
                tactile_cnn_strides=tuple(cnn_strides),
                tactile_cnn_padding=cnn_padding,
                mask_touch=mask_touch,
            )
        else:
            value_def = PixelMultiplexer(
                encoder=encoder,
                network=value_head,
                latent_dim=latent_dim,
                use_bottleneck=use_bottleneck,
            )

        variables = value_def.init(
            jax.random.PRNGKey(seed), observations)
        tx = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate),
        )
        self._state = TrainState.create(
            apply_fn=value_def.apply,
            params=variables["params"],
            tx=tx,
            batch_stats=variables.get("batch_stats"),
        )
        self._loss_coefficient = float(value_loss_coefficient)

    def value(self, observations) -> np.ndarray:
        return np.asarray(jax.device_get(_value_jit(
            self._state, observations)))

    def update(self, observations, returns) -> dict[str, float]:
        self._state, info = _update_value_jit(
            self._state, observations, returns, self._loss_coefficient)
        return {
            key: float(np.asarray(jax.device_get(value)))
            for key, value in info.items()
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "step": self._state.step,
            "params": self._state.params,
            "opt_state": self._state.opt_state,
            "batch_stats": self._state.batch_stats,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._state = self._state.replace(
            step=state["step"],
            params=state["params"],
            opt_state=state["opt_state"],
            batch_stats=state["batch_stats"],
        )
