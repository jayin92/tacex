"""Isolated MaxInfoSAC baseline using summed-taxel prediction uncertainty."""

from jaxrl2.agents.pixel_maxinfosac.pixel_maxinfosac_learner import (
    PixelMaxinfoSACLearner,
)
from jaxrl2.agents.pixel_maxinfosac_taxel_sum.networks import (
    PixelMaxInfoTaxelSumCritic,
)


class PixelMaxinfoSACTaxelSumLearner(PixelMaxinfoSACLearner):
    """MaxInfoSAC with summed-taxel rather than tactile-map uncertainty."""

    critic_cls = PixelMaxInfoTaxelSumCritic
