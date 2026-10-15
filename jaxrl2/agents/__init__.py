from jaxrl2.agents.pixel_sac import PixelSACLearner
from jaxrl2.agents.pixel_maxinfosac import PixelMaxinfoSACLearner
from jaxrl2.agents.pixel_maxinfosac_explorer import PixelMaxinfoSACExplorer

# The imported exploration learners use the optional Gymnasium jaxrl fork.
# Resolve them only when requested, so diffusion steering does not require it.
_EXPLORATION_LEARNERS = {
    "MaxInfoSacLearner": "maxinfosac.maxinfosac_learner",
    "MaxInfoMBSacLearner": "maxinfombsac.maxinfombsac_learner",
    "DrQv2Learner": "drqv2.drqv2_learner",
    "MaxInfoDrQv2Learner": "maxinfodrqv2.maxinfodrqv2_learner",
    "MaxInfoMBDrQv2Learner": "maxinfombdrqv2.maxinfombdrqv2_learner",
    "MaxInfoDrQLearner": "maxinfodrq.maxinfodrq_learner",
    "DrQExplorerLearner": "drq_explorer.drq_explorer_learner",
    "SkillSACLearner": "skillsac.skillsac_learner",
    "TactileDrQLearner": "tactile_agents.tactile_drq.tactile_drq_learner",
    "TactileDrQv2Learner": "tactile_agents.tactile_drqv2.tactile_drqv2_learner",
    "TactileMaxInfoDrQLearner": "tactile_agents.tactile_maxinfodrq.tactile_maxinfodrq_learner",
    "TactileMaxInfoDrQv2Learner": "tactile_agents.tactile_maxinfodrqv2.tactile_maxinfodrqv2_learner",
    "TactileDrQExplorer": "tactile_agents.tactile_explorer.drq_explorer_learner",
}


def __getattr__(name):
    if name not in _EXPLORATION_LEARNERS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    try:
        learner = getattr(import_module(f"{__name__}.{_EXPLORATION_LEARNERS[name]}"), name)
    except ModuleNotFoundError as exc:
        if exc.name and (exc.name == "jaxrl" or exc.name.startswith("jaxrl.")):
            raise ModuleNotFoundError(
                "Exploration learners require the Gymnasium jaxrl fork. "
                "Install it with: python -m pip install -e '.[exploration]'"
            ) from exc
        raise
    globals()[name] = learner
    return learner
