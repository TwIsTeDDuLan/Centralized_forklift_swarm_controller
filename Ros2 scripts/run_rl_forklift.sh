#!/usr/bin/env bash
# Run the RL go-to-pose controller for one or more forklifts (same topics as run_forklift_N.sh).
#   ./run_rl_forklift.sh 1            # forklift 1
#   ./run_rl_forklift.sh 1 2 3        # all three, in the background of this shell; Ctrl+C stops all
#   ./run_rl_forklift.sh 1 --ros-args -p px_per_m:=65.0
# Policy: policies/forklift_nav.npz (from isaac_lab_forklift/scripts/export_policy.py).
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -z "${ROS_DISTRO:-}" ]] && [[ -f /opt/ros/jazzy/setup.bash ]]; then
    set +eu
    source /opt/ros/jazzy/setup.bash
    set -eo pipefail
fi

if [[ ! -f policies/forklift_nav.npz ]]; then
    echo "[ERROR] policies/forklift_nav.npz not found; run isaac_lab_forklift/scripts/export_policy.py first" >&2
    exit 1
fi

IDS=()
while [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; do IDS+=("$1"); shift; done
[[ ${#IDS[@]} -eq 0 ]] && IDS=(1)

export PYTHONUNBUFFERED=1
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
