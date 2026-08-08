"""Stationary AI spatula-lift manager env (teacher variant).

The thin-object flagship: the LBM wooden flat spatula (66 g, 30.0 x 7.0 x 5.1 cm,
same asset as the G1 spatula task) replaces the DexCube on the proven Stationary AI
cube-lift scaffold. Everything debugged there is inherited unchanged -- physics
presets (PhysX / Newton MJWarp / Newton adaptive), the no-rails rig, EE TCP offset,
obs groups, goal/reset bands, PPO wiring.

GRASP GEOMETRY (measured, decides everything): the WXAI gripper's carriage travel is
0 -> 0.044 m with a HARD lower limit at 0, and the closed-finger gap is 4.83 cm
(diag_grasp_geom.py, see cube task). The spatula HANDLE (~2.2 cm) is therefore
ungraspable with the official model -- the fingers close past it. The BLADE is
6.98 cm wide: gripping ACROSS the blade gives 2.15 cm of squeeze, dimensionally the
same pinch as the proven 5.4 cm cube (0.57 cm squeeze). So this task is
lift-by-the-blade, and that is the point: finger pads clamping a millimeters-thick
wooden plate resting on a rigid tabletop is exactly the stiff thin-object contact
regime where fixed-step integration artifacts are largest.

The asset's body frame has X along the length with the origin at the blade/handle
junction (blade x in [-0.053, 0.057], handle out to x=0.247), so the stock lift
reward's object-ROOT reach target already lands on the blade -- no custom reach
term is needed for v0.
"""

from __future__ import annotations

import os

from trossen_cube.paths import STATIONARY_AI_USD

from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.schemas import RigidBodyPropertiesCfg
from isaaclab.sim.spawners import UsdFileCfg
from isaaclab.utils.configclass import configclass

from ..cube_lift.cube_lift_env_cfg import StationaryAiCubeLiftEnvCfg

SPATULA_USD_PATH = os.path.join(os.path.dirname(__file__), "assets", "thimma_wood_natural_flat_spatula.usd")
"""Converted LBM spatula, identical file to the G1 task's (separate blade and handle
collision prims; body frame per the Drake SDF)."""

# Rig tabletop slab top is z=0.02 (see cube task). The blade lies flat with its root
# ~3 mm above the underside; spawn just above and let the reset settle it.
SPATULA_REST_Z = 0.025

# Root rest height is ~0.023 m, so 0.09 (inherited from the cube task) would demand a
# ~6.7 cm lift -- fine -- but the blade-gripped spatula tilts handle-down when lifted,
# which lowers the ROOT relative to the grip point. 0.08 keeps success = unambiguous
# lift-off without punishing the tilt.
LIFT_HEIGHT = 0.08


@configclass
class StationaryAiSpatulaLiftEnvCfg(StationaryAiCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Manipuland: the spatula, lying flat, handle pointing +x (across the arm's
        # front), blade at the origin where the proven cube reset/goal bands already
        # guarantee reachability. No yaw randomization in v0: the blade must cross the
        # finger axis for the pinch to seat, and the cube task's reset has no yaw
        # either -- revisit once the grasp is reliable.
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.13, SPATULA_REST_Z], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=SPATULA_USD_PATH,
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Same reset convention as the cube task (z jitter 0): tabletop objects start
        # resting. Dropping a 66 g thin blade from height flips it airborne before the
        # policy ever acts.
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.10, 0.10),
            "y": (-0.075, 0.075),
            "z": (0.0, 0.0),
        }

        self.rewards.lifting_object.params["minimal_height"] = LIFT_HEIGHT
        self.rewards.object_goal_tracking.params["minimal_height"] = LIFT_HEIGHT
        self.rewards.object_goal_tracking_fine_grained.params["minimal_height"] = LIFT_HEIGHT

        # TROSSEN_RAILS=1: contact-rich ablation. The cube task disabled the rig's rail
        # frame because the policy learned to JAM the object against it instead of
        # grasping -- expect the same exploit pressure here (a thin blade wedges even
        # more easily than a cube), which is why v0 trains no-rails. With rails on, the
        # scene gains the full frame's collision surfaces: a harder, more contact-dense
        # variant for the adaptive-vs-fixed comparison once the clean task works.
        if os.environ.get("TROSSEN_RAILS") == "1":
            self.scene.robot.spawn.usd_path = STATIONARY_AI_USD


@configclass
class StationaryAiSpatulaLiftEnvCfg_PLAY(StationaryAiSpatulaLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
