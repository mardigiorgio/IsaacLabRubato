# IsaacLabRubato

IsaacLabRubato is a modified [Isaac Sim](https://developer.nvidia.com/isaac/sim) /
[Isaac Lab](https://github.com/isaac-sim/IsaacLab) / [Newton](https://github.com/newton-physics/newton)
stack: the standard Isaac toolchain with an adaptive-timestepping solver wired into Isaac Lab's Newton
backend as a selectable option, so reinforcement-learning policies can be trained on adaptive-step
(rather than fixed-step) physics. It is the Isaac stack plus that integration, not a standalone platform.

*Rubato* (It., "stolen") is the flexible variation of local tempo in music. Here it names adaptive
timestepping, where integration effort is reallocated across a frame (spent at stiff contact, withheld in
free motion) while the frame, and with it the fixed-rate control boundary, is preserved.

## How it works

Isaac Lab steps physics through a single manager class (`NewtonMJWarpManager`, in its `isaaclab_newton`
extension) that builds and steps the Newton solver each control interval. The integration plugs into that
class's solver-construction and solver-stepping seams, so the adaptive solver (`SolverMuJoCoAdaptive`,
error-controlled step-doubling over MuJoCo-Warp) is selectable alongside PhysX and stock Newton, with the
policy, rewards, observations, and rendering unchanged. The control interface is preserved exactly: the
action is zero-order-held across the interval, and the solver subdivides time only *within* the interval,
always landing on the boundary.

## Installation

Prerequisites you install yourself: [`uv`](https://docs.astral.sh/uv/getting-started/installation/),
an NVIDIA driver >= 580, `git`, and ~60 GB free disk. Then:

```bash
git clone https://github.com/mardigiorgio/IsaacLabRubato.git && cd IsaacLabRubato
bash install/install.sh
```

The script is idempotent and ends by running `install/verify.py`, which confirms the Newton fork is the
active import, `SolverMuJoCoAdaptive` (and, when `sap_warp` is present, the SAP variants) are available,
and the Isaac Lab fork carries the `--solver` wiring and the `newton_adaptive_ui` extension.

<details>
<summary>What install.sh does (the manual steps)</summary>

```bash
# clone the custom Newton fork, the SAP solver, and the Isaac Lab fork as SIBLINGS of this repo
# (the fork's `develop` branch already carries the Newton + SAP integration -- no patch step)
git clone https://github.com/mardigiorgio/newton-adaptive.git ../newton-adaptive
git clone https://github.com/mardigiorgio/sap_warp.git ../sap_warp
git clone -b develop https://github.com/mardigiorgio/IsaacLab.git ../IsaacLab

# sap_warp has no installable package: the Newton fork adds its root to sys.path at import
# time, probing $SAP_WARP_PATH and falling back to the sibling clone. Export SAP_WARP_PATH
# only when sap_warp is NOT at the default sibling path.

# build the venv and install the locked platform (Isaac Sim + PyTorch cu128 + the Newton fork)
uv venv --python 3.12 .venv
uv sync --locked

# install Isaac Lab's editable extensions into the same venv, then re-assert the fork
# over Isaac Lab's stock git-pinned Newton (isaaclab.sh -i installs the stock pin)
VIRTUAL_ENV="$PWD/.venv" OMNI_KIT_ACCEPT_EULA=YES ../IsaacLab/isaaclab.sh -i
uv sync --inexact --locked

# verify
uv run python install/verify.py
```

</details>

Update the Newton fork later with `git -C ../newton-adaptive pull && uv sync --inexact --locked`;
pull Isaac Lab changes with `git -C ../IsaacLab pull origin develop` followed by
`VIRTUAL_ENV="$PWD/.venv" OMNI_KIT_ACCEPT_EULA=YES ../IsaacLab/isaaclab.sh -i && uv sync --inexact --locked`
(re-running `-i` refreshes the editable extensions and their dependencies; the final sync
re-asserts the fork).

## Getting started

Activate the platform venv first, so `isaaclab.sh` and the launcher use it (without this, a fresh
install falls back to the wrong Python):

```bash
source .venv/bin/activate     # Isaac Sim + Isaac Lab + the Newton fork
```

**Editor** - build/inspect scenes on the Newton backend (from the repo root):

```bash
./isaaclab-rubato
```

**Smoke test** - a 5-iteration cartpole training run on the adaptive solver, headless:

```bash
OMNI_KIT_ACCEPT_EULA=YES ../IsaacLab/isaaclab.sh train --rl_library rsl_rl \
  --task Isaac-Cartpole-Direct --num_envs 16 presets=newton_mjwarp \
  --solver mujoco-adaptive --max_iterations 5
```

(`OMNI_KIT_ACCEPT_EULA=YES` matters only for the very first Isaac Sim launch on a
machine; the sweep driver and the editor launcher set it themselves.)

**Training studies** - the fixed-vs-adaptive PPO sweep lives in `experiments/rubato-ppo-sweep/`:

```bash
cd experiments/rubato-ppo-sweep
TASKS="Isaac-Velocity-Flat-G1-v0" SOLVERS="mujoco mujoco-adaptive" SEEDS="42 43 44" bash sweep.sh
```

`sweep.sh` is env-var driven (`TASKS`, `SOLVERS`, `SEEDS`, `VIDEO=1`, `WANDB_MODE=offline`, ...);
see its header for the full knob list. Earlier one-off studies remain under
`experiments/06-30-2026-experiments/` and `experiments/g1_dish_rack/` as records.

### Selecting the solver (`--solver`)

The Newton backend exposes four solver variants through the `--solver` flag on the `isaaclab.sh
train` entry point (`source/isaaclab_rl/isaaclab_rl/entrypoints/backends/train_rsl_rl.py`, baked into
the IsaacLab fork's `develop` branch; the mimic and teleop scripts expose the same flag). It drives the
`MJWarpSolverCfg` latches (`backend` / `adaptive` / `sap_adaptive`) read by `NewtonMJWarpManager`:

| `--solver` | backend | constructs | notes |
|---|---|---|---|
| `mujoco` | MuJoCo-Warp | `SolverMuJoCo` | fixed-step (stock Newton default) |
| `mujoco-adaptive` | MuJoCo-Warp | `SolverMuJoCoAdaptive` | error-controlled step-doubling |
| `sap` | SAP (`sap_warp`) | `SolverSAP` | fixed-step convex compliant contact |
| `sap-adaptive` | SAP (`sap_warp`) | `SolverSAPAdaptive` | step-doubling SAP (even + global tiling) |

The two `sap*` variants require the `sap_warp` clone (sibling dir or `SAP_WARP_PATH`; see install).
Run the built-in cube-reorient study task (Newton-tested: Allegro hand) with, e.g.:

```bash
../IsaacLab/isaaclab.sh train --solver sap-adaptive \
  --task Isaac-Reorient-Cube-Allegro-Direct \
  --rl_library rsl_rl --headless --num_envs 64 --max_iterations 1 presets=newton_mjwarp
# swap --solver for mujoco | mujoco-adaptive | sap | sap-adaptive
```

The adaptive paths (`mujoco-adaptive`, `sap-adaptive`) also respond to the config flag
`MJWarpSolverCfg(adaptive=True)`, the `NEWTON_ADAPTIVE=1` env var, and the GUI toggle (the
`newton_adaptive_ui` Kit extension in the IsaacLab fork); `NEWTON_ADAPTIVE_LOG_EVERY=N` writes
per-frame dt + sub-step counts to `/tmp/newton_adaptive.log`.

## Built on

Layered on three upstream projects, used under their own licenses:

- **Isaac Sim**: NVIDIA Omniverse robotics simulator. <https://developer.nvidia.com/isaac/sim>
- **Isaac Lab**: NVIDIA RL / robot-learning framework (BSD-3-Clause). <https://github.com/isaac-sim/IsaacLab>
- **Newton**: GPU physics over MuJoCo-Warp (Apache-2.0). <https://github.com/newton-physics/newton>

This repository contributes the adaptive-solver integration, the RL workstream, the scenes, and the
install automation.
