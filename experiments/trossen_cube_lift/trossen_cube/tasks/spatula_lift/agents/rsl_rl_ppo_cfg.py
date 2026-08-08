"""PPO runner for the spatula-lift teacher: the cube-lift recipe, own experiment name.

Identical hyperparameters to the cube task by design -- the flagship comparison is
fixed-vs-adaptive physics on the SAME task and recipe, so nothing here should drift
from the proven cube configuration except where the object demands it.
"""

from isaaclab.utils.configclass import configclass

from ...cube_lift.agents.rsl_rl_ppo_cfg import StationaryAILiftPPORunnerCfg


@configclass
class StationaryAISpatulaLiftPPORunnerCfg(StationaryAILiftPPORunnerCfg):
    experiment_name = "stationary_ai_spatula_teacher"
