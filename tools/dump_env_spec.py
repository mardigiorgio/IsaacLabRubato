"""Dump the Isaac-Velocity-Flat-G1 environment spec to JSON.

Instantiates the env exactly the way rubato-ppo-sweep's training did (same
launcher args, same hydra preset tokens such as ``physics=newton_mjwarp`` and
``--solver mujoco``) and records everything a standalone deployment needs:
joint order, default joint positions, PD gains, action scale/offset, obs term
layout, and control rates.

Run from anywhere with the isaac-rubato venv active:

    ISAACLAB=~/Documents/code/IsaacLab
    $ISAACLAB/isaaclab.sh -p policy/dump_env_spec.py \
        --task Isaac-Velocity-Flat-G1 --solver mujoco --headless \
        --out env_spec.json physics=newton_mjwarp
"""

import argparse
import json
import sys

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.string import list_intersection

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

# same latch map as integration/cli_solver_flag.patch
_SOLVER_CHOICES = {
    "mujoco": {"backend": "mujoco", "adaptive": False, "sap_adaptive": False},
    "mujoco-adaptive": {"backend": "mujoco", "adaptive": True, "sap_adaptive": False},
    "sap": {"backend": "sap", "adaptive": False, "sap_adaptive": False},
    "sap-adaptive": {"backend": "sap", "adaptive": False, "sap_adaptive": True},
}

parser = argparse.ArgumentParser(description="Dump env spec for deployment.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--out", type=str, default="env_spec.json")
parser.add_argument("--solver", default=None, choices=sorted(_SOLVER_CHOICES))
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + list_intersection(remaining_args, None)

OUT_PATH = args_cli.out


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    with launch_simulation(env_cfg, args_cli):
        import gymnasium as gym

        if args_cli.solver is not None:
            solver_cfg = env_cfg.sim.physics.solver_cfg
            for field, value in _SOLVER_CHOICES[args_cli.solver].items():
                setattr(solver_cfg, field, value)
            print(f"[dump_env_spec] --solver={args_cli.solver} applied")

        env_cfg.scene.num_envs = args_cli.num_envs
        env = gym.make(args_cli.task, cfg=env_cfg)
        uenv = env.unwrapped
        robot = uenv.scene["robot"]

        obs, _ = env.reset()

        obs_mgr = uenv.observation_manager
        act_mgr = uenv.action_manager
        act_term = act_mgr.get_term("joint_pos")

        spec = {
            "task": args_cli.task,
            "physics_dt": float(uenv.physics_dt),
            "decimation": int(uenv.cfg.decimation),
            "step_dt": float(uenv.step_dt),
            "joint_names": list(robot.joint_names),
            "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
            "joint_stiffness": robot.data.joint_stiffness[0].cpu().tolist(),
            "joint_damping": robot.data.joint_damping[0].cpu().tolist(),
            "joint_armature": robot.data.joint_armature[0].cpu().tolist(),
            "joint_effort_limits": robot.data.joint_effort_limits[0].cpu().tolist(),
            "joint_pos_limits": robot.data.joint_pos_limits[0].cpu().tolist(),
            "default_root_state": robot.data.default_root_state[0].cpu().tolist(),
            "action_scale": act_term._scale.cpu().tolist()
            if hasattr(act_term._scale, "cpu")
            else act_term._scale,
            "action_offset": act_term._offset[0].cpu().tolist()
            if hasattr(act_term._offset, "cpu")
            else act_term._offset,
            "obs_terms_policy": list(obs_mgr.active_terms["policy"]),
            "obs_term_dims_policy": [
                list(d) if hasattr(d, "__iter__") else int(d)
                for d in obs_mgr.group_obs_term_dim["policy"]
            ],
            "obs_dim_total": int(obs["policy"].shape[-1]),
            "num_actions": int(act_mgr.total_action_dim),
            "body_names": list(robot.body_names),
        }

        with open(OUT_PATH, "w") as f:
            json.dump(spec, f, indent=2)
        print(f"[dump_env_spec] wrote {OUT_PATH}")

        # dump the exact MuJoCo model the Newton MJWarp solver compiled from
        # the USD: this is the physics description training actually stepped
        try:
            import mujoco

            from isaaclab_newton.physics.newton_manager import NewtonManager

            solver = NewtonManager._solver
            mj_model = getattr(solver, "mj_model", None)
            if mj_model is not None:
                mjb_path = OUT_PATH.replace(".json", "_model.mjb")
                mujoco.mj_saveModel(mj_model, mjb_path, None)
                jnames = [
                    mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j)
                    for j in range(mj_model.njnt)
                ]
                anames = [
                    mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
                    for a in range(mj_model.nu)
                ]
                print(f"[dump_env_spec] wrote {mjb_path} "
                      f"(njnt={mj_model.njnt}, nu={mj_model.nu})")
                print(f"[dump_env_spec] mj joints: {jnames}")
                print(f"[dump_env_spec] mj actuators: {anames}")
            else:
                print("[dump_env_spec] solver has no mj_model; skipped MJCF dump")
        except Exception as e:  # noqa: BLE001
            print(f"[dump_env_spec] MJCF dump failed: {e}")
        print(json.dumps({k: spec[k] for k in ("joint_names", "obs_terms_policy", "obs_dim_total")}, indent=2))

        env.close()


if __name__ == "__main__":
    main()
