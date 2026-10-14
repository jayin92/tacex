"""Isolated SAC baseline rewarded by aggregate tactile contact magnitude."""

from typing import Dict

import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
from jaxrl2.agents.taxel_utils import summed_taxel_reward


class PixelSACTaxelRewardLearner(PixelSACLearner):
    """Pixel SAC trained only on the next observation's summed taxel signal.

    The replay buffer retains the sparse task reward for evaluation and success
    reporting.  Only the batch passed to SAC's update is relabelled, keeping
    collectors and every existing experiment unchanged.
    """

    taxel_reward_scale = 1.0

    def update(self, batch: FrozenDict) -> Dict[str, jnp.ndarray]:
        task_rewards = jnp.asarray(batch["rewards"])
        taxel_rewards = summed_taxel_reward(
            batch, scale=self.taxel_reward_scale)
        relabelled_batch = batch.copy(
            add_or_replace={"rewards": taxel_rewards})

        info = super().update(relabelled_batch)
        return {
            **info,
            "task_reward_mean": task_rewards.mean(),
            "taxel_reward_scale": jnp.asarray(self.taxel_reward_scale),
            "taxel_reward_mean": taxel_rewards.mean(),
            "taxel_reward_std": taxel_rewards.std(),
            "taxel_reward_min": taxel_rewards.min(),
            "taxel_reward_max": taxel_rewards.max(),
        }
