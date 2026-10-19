#!/usr/bin/env bash
# ROS side of the warehouse loop. Isaac Sim stays manual:
#   1. Open Issasim Models/Environment.usd in Isaac Sim and press Play
#   2. ./run_warehouse.sh              # tracker + RL forklifts 1 2 3
#   3. Attach fork lidar (./run_fork_lidar.sh) if you want insert, not just nav
#   4. Copy a pub from cmds.txt
#
#   ./run_warehouse.sh --same          # both in this terminal
#   ./run_warehouse.sh --no-tracker    # RL nodes only
#   ./run_warehouse.sh --fleet         # also open the fleet prompt (pallet 1 1)
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
chmod +x run_tracker.sh run_rl_forklift.sh run_fleet_controller.sh run_fork_lidar.sh 2>/dev/null || true

LAUNCH_TRACKER=1
LAUNCH_RL=1
LAUNCH_FLEET=0
SAME=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --same|--same-term) SAME=1; shift ;;
        --no-tracker) LAUNCH_TRACKER=0; shift ;;
        --fleet) LAUNCH_FLEET=1; shift ;;
        -h|--help)
            sed -n '2,12p' "$0"
            exit 0
            ;;
        *)
            echo "unknown option: $1" >&2
            exit 1
            ;;
    esac
done

echo "Isaac Sim must already be playing Environment.usd."
echo "After these nodes are up, send a goal from cmds.txt"
echo "  (or ./run_fork_lidar.sh, then 'pallet 1 1' in the fleet prompt)."
echo ""

if [[ "${SAME}" -eq 1 ]]; then
    PIDS=()
    cleanup() { kill "${PIDS[@]}" 2>/dev/null || true; wait 2>/dev/null || true; }
    trap cleanup SIGINT SIGTERM
    [[ ${LAUNCH_TRACKER} -eq 1 ]] && { ./run_tracker.sh & PIDS+=($!); }
    [[ ${LAUNCH_RL} -eq 1 ]] && { ./run_rl_forklift.sh 1 2 3 & PIDS+=($!); }
    [[ ${LAUNCH_FLEET} -eq 1 ]] && { ./run_fleet_controller.sh & PIDS+=($!); }
    wait
    exit 0
fi

# Reuse run_all's window launcher with RL forklifts (no hand-tuned nodes).
ARGS=(--windows)
[[ ${LAUNCH_TRACKER} -eq 1 ]] || ARGS+=(--only-forklifts)
[[ ${LAUNCH_FLEET} -eq 1 ]] || ARGS+=()
if [[ ${LAUNCH_TRACKER} -eq 1 && ${LAUNCH_FLEET} -eq 0 ]]; then
    # tracker + RL, skip fleet
    exec ./run_all.sh --rl --windows --no-fleet
fi
if [[ ${LAUNCH_TRACKER} -eq 0 && ${LAUNCH_FLEET} -eq 0 ]]; then
    exec ./run_rl_forklift.sh 1 2 3
fi
exec ./run_all.sh --rl --windows
