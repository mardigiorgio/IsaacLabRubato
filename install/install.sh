#!/usr/bin/env bash
# IsaacLabRubato installer -- one command to a verified platform venv.
#
# Idempotent: re-running reuses existing clones and the existing venv.
# Layout contract: this repo, IsaacLab, newton-adaptive, and sap_warp are
# SIBLING directories. Paths are computed from this script's own location,
# so the repo directory may be named or moved freely.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$(dirname "$ROOT")"

# --- preflight --------------------------------------------------------------
command -v uv >/dev/null || { echo "ERROR: uv not found -- install it first: https://docs.astral.sh/uv/getting-started/installation/" >&2; exit 1; }
command -v git >/dev/null || { echo "ERROR: git not found" >&2; exit 1; }
if command -v nvidia-smi >/dev/null 2>&1; then
    driver_major="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)"
    if [ "${driver_major:-0}" -lt 580 ]; then
        echo "WARNING: NVIDIA driver ${driver_major:-unknown} < 580 -- Isaac Sim 6 requires >= 580." >&2
    fi
else
    echo "WARNING: nvidia-smi not found -- an NVIDIA GPU with driver >= 580 is required at runtime." >&2
fi
free_gb="$(df -B1G --output=avail "$CODE_DIR" 2>/dev/null | tail -1 | tr -d ' ' || echo 0)"
if [ "${free_gb:-0}" -lt 60 ]; then
    echo "WARNING: ~60 GB free disk recommended for the full platform; ${free_gb:-unknown} GB available." >&2
fi

# --- sibling clones ---------------------------------------------------------
# The IsaacLab fork's develop branch carries the whole Newton/SAP delta as
# real commits -- there is no patch step.
[ -d "$CODE_DIR/newton-adaptive" ] || git clone https://github.com/mardigiorgio/newton-adaptive.git "$CODE_DIR/newton-adaptive"
[ -d "$CODE_DIR/sap_warp" ] || git clone https://github.com/mardigiorgio/sap_warp.git "$CODE_DIR/sap_warp"
[ -d "$CODE_DIR/IsaacLab" ] || git clone -b develop https://github.com/mardigiorgio/IsaacLab.git "$CODE_DIR/IsaacLab"

# --- venv + locked platform (Isaac Sim + torch cu128 + editable Newton fork) -
cd "$ROOT"
[ -d .venv ] || uv venv --python 3.12 .venv
uv sync --locked

# --- Isaac Lab editable extensions into the same venv -----------------------
# This step also installs Isaac Lab's own git-pinned Newton over the fork ...
VIRTUAL_ENV="$ROOT/.venv" OMNI_KIT_ACCEPT_EULA=YES "$CODE_DIR/IsaacLab/isaaclab.sh" -i

# --- ... so re-assert the editable Newton fork (and its mujoco 3.8 stack) ----
uv sync --inexact --locked

# --- SAP solver path --------------------------------------------------------
# The Newton fork probes $SAP_WARP_PATH and falls back to the sibling
# sap_warp clone; export SAP_WARP_PATH only for non-sibling layouts.
[ -d "$CODE_DIR/sap_warp" ] || echo "NOTE: sap_warp clone missing -- the sap / sap-adaptive solver variants will be disabled." >&2

# --- verify -----------------------------------------------------------------
uv run python "$ROOT/install/verify.py"
echo "Install complete. Activate with:  source $ROOT/.venv/bin/activate"
