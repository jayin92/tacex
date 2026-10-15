import functools
from typing import Sequence, Callable

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from jaxrl.agents.drq.networks import Encoder
from jaxrl.networks.policies import MSEPolicy
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class DrQv2Policy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    cnn_features: Sequence[int] = (32, 32, 32, 32)
    cnn_strides: Sequence[int] = (2, 1, 1, 1)
    cnn_padding: str = 'VALID'
    latent_dim: int = 50

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0) -> tfd.Distribution:
        x = Encoder(self.cnn_features,
                    self.cnn_strides,
                    self.cnn_padding,
                    name='SharedEncoder')(observations)

        # We do not update conv layers with policy gradients.
        x = jax.lax.stop_gradient(x)

        x = nn.Dense(self.latent_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)

        return MSEPolicy(self.hidden_dims, self.action_dim)(x, temperature)


class DrQv2PolicyDormantRatioCalculator(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    cnn_features: Sequence[int] = (32, 32, 32, 32)
    cnn_strides: Sequence[int] = (2, 1, 1, 1)
    cnn_padding: str = 'VALID'
    latent_dim: int = 50
    percentage: float = 0.025
    activate_final: bool = True
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    compute_after_activations: bool = True

    @nn.compact
    def __call__(self, observations: jnp.ndarray):
        x = Encoder(self.cnn_features,
                    self.cnn_strides,
                    self.cnn_padding,
                    name='SharedEncoder')(observations)

        x = nn.Dense(self.latent_dim)(x)

        # look at average activation over the batch
        dormant_neurons = 0
        num_neurons = 0
        # mean over batch
        abs_x = jnp.abs(x).mean(axis=0)
        # compute number of neurons which have low activation
        H_tau = (abs_x < (abs_x.mean() * self.percentage)).sum()
        dormant_neurons += H_tau
        # Total num neurons in the layer
        num_neurons += self.latent_dim

        x = nn.LayerNorm()(x)
        x = nn.tanh(x)

        # going over the MSEPolicy
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size)(x)
            # computing dormant ratio after the relu activation (generally results in larger values)
            if self.compute_after_activations:
                if i + 1 < len(self.hidden_dims) or self.activate_final:
                    x = self.activations(x)
                # mean over batch
                abs_x = jnp.abs(x).mean(axis=0)
                # compute number of neurons which have low activation
                H_tau = (abs_x < (abs_x.mean() * self.percentage)).sum()
            else:
                # mean over batch
                abs_x = jnp.abs(x).mean(axis=0)
                # compute number of neurons which have low activation
                H_tau = (abs_x < (abs_x.mean() * self.percentage)).sum()
                if i + 1 < len(self.hidden_dims) or self.activate_final:
                    x = self.activations(x)
            dormant_neurons += H_tau
            # Total num neurons in the layer
            num_neurons += size

        x = nn.Dense(self.action_dim)(x)
        # mean over batch
        abs_x = jnp.abs(x).mean(axis=0)
        # compute number of neurons which have low activation
        H_tau = (abs_x < (abs_x.mean() * self.percentage)).sum()
        dormant_neurons += H_tau
        # Total num neurons in the layer
        num_neurons += self.action_dim
        beta = dormant_neurons / num_neurons
        return beta

    @functools.partial(jax.jit, static_argnums=0)
    def convert_params(self, drqv2_policy_params: PyTree):
        converted_params = {}
        converted_params['SharedEncoder'] = drqv2_policy_params['SharedEncoder'].copy()
        converted_params['Dense_0'] = drqv2_policy_params['Dense_0'].copy()
        converted_params['LayerNorm_0'] = drqv2_policy_params['LayerNorm_0'].copy()
        i = 0
        for i, size in enumerate(self.hidden_dims):
            converted_params[f'Dense_{i + 1}'] = drqv2_policy_params['MSEPolicy_0']['MLP_0'][f'Dense_{i}'].copy()

        converted_params[f'Dense_{i + 2}'] = drqv2_policy_params['MSEPolicy_0']['Dense_0'].copy()
        return converted_params


@jax.jit
def perturb_params(init_params: PyTree,
                   trained_params: PyTree,
                   perturb_factor: chex.Array,
                   perturb_keys: PyTree,
                   ):
    # if z = 1.0, we perturb the params, else z = 0.0 and we keep the trained params
    perturbation_fn = lambda x, y, z: z * (x * perturb_factor + (1 - perturb_factor) * y) + (1 - z) * x
    new_params = jax.tree_util.tree_map(perturbation_fn,
                                        trained_params,
                                        init_params,
                                        perturb_keys
                                        )
    return new_params
