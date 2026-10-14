"""MaxInfoSAC critic whose tactile dynamics target is summed taxel magnitude."""

from typing import Dict, Tuple

import flax.linen as nn
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from jaxrl2.agents.pixel_maxinfosac.networks import PixelMaxInfoCritic
from jaxrl2.agents.taxel_utils import summed_taxel_magnitude


class PixelMaxInfoTaxelSumCritic(PixelMaxInfoCritic):
    """Keep full tactile-map Q features but model one aggregate signal."""

    @nn.compact
    def __call__(
            self,
            observations: Dict[str, jnp.ndarray],
            actions: jnp.ndarray,
            training: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, FrozenDict]:
        critic, expl_critic, latent, modalities = super().__call__(
            observations, actions, training=training)
        if "tactile" not in observations:
            raise ValueError(
                "PixelMaxInfoSAC taxel-sum exploration requires a tactile "
                "observation. Launch it with add_tactile=1 and use_touch=1.")
        modalities = modalities.copy(add_or_replace={
            "tactile": summed_taxel_magnitude(observations["tactile"]),
        })
        return critic, expl_critic, latent, modalities
