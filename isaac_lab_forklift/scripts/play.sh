#!/usr/bin/env bash
# Watch the latest trained policy (with a GUI) and export it.
#   ./scripts/play.sh                                  # latest run, latest checkpoint, 16 envs
#   ./scripts/play.sh --load_run <run_dir_name> --checkpoint model_500.pt
# Isaac Lab's play script also writes <run>/exported/policy.pt + policy.onnx; then run
# scripts/export_policy.py to get the numpy weights the ROS node loads.
set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB=${ISAACLAB:-$HOME/IsaacLab}
TASK=${TASK:-Isaac-Forklift-Nav-Direct}
cd "$HERE"
# isaaclab.sh -p puts Isaac Sim on the path; it needs the venv active
source "$ISAACLAB/.venv/bin/activate"
exec "$ISAACLAB/isaaclab.sh" -p "$ISAACLAB/scripts/reinforcement_learning/play.py" \
    --rl_library rsl_rl --task "$TASK" --num_envs 16 --viz kit "$@"
