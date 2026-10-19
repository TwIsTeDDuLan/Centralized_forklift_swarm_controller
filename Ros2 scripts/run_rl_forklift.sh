#!/usr/bin/env bash
# Two-policy RL controllers (nav to staging, then insert). Isaac Sim must already be playing.
#   ./run_rl_forklift.sh              # forklifts 1 2 3 (default)
#   ./run_rl_forklift.sh 1            # just forklift 1
#   ./run_rl_forklift.sh 1 2 3 --ros-args -p auto_insert:=true
# Policies: policies/forklift_nav.npz and policies/forklift_insert.npz
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -z "${ROS_DISTRO:-}" ]] && [[ -f /opt/ros/jazzy/setup.bash ]]; then
    set +eu
    source /opt/ros/jazzy/setup.bash
    set -eo pipefail
fi

if [[ ! -f policies/forklift_nav.npz ]]; then
    echo "[ERROR] policies/forklift_nav.npz not found; from isaac_lab_forklift/ run:" >&2
    echo "  ~/IsaacLab/.venv/bin/python scripts/export_policy.py" >&2
    exit 1
fi
if [[ ! -f policies/forklift_insert.npz ]]; then
    echo "[WARN] policies/forklift_insert.npz missing; nodes will stop at staging (no insert)." >&2
fi

IDS=()
while [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; do IDS+=("$1"); shift; done
[[ ${#IDS[@]} -eq 0 ]] && IDS=(1 2 3)

export PYTHONUNBUFFERED=1

echo "RL forklifts ${IDS[*]}  (Ctrl+C stops all)"
echo "  nav:    policies/forklift_nav.npz"
echo "  insert: policies/forklift_insert.npz"
echo "  goals:  /forklift_N/lift_target_pos   (see cmds.txt)"
echo "  insert also needs /forklift_N/scan    (./run_fork_lidar.sh after the scene is playing)"
echo ""

if [[ ${#IDS[@]} -eq 1 ]]; then
    exec python3 rl_forklift_controller.py "${IDS[0]}" "$@"
fi

PIDS=()
for id in "${IDS[@]}"; do
    python3 rl_forklift_controller.py "$id" "$@" &
    PIDS+=($!)
done
trap 'kill "${PIDS[@]}" 2>/dev/null' SIGINT SIGTERM
wait
