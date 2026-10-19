#!/usr/bin/env bash
# Pick -> deliver -> place test without Isaac Sim: tests/fake_world.py (kinematic forklift 1, pallet 1,
# camera, mast and fork lidar using the insert policy's training pallet model) + the real RL node, driven by
# the fleet controller's commands. Uses ROS_DOMAIN_ID 81 so it cannot disturb a running warehouse.
#   ./run_fake_pick_test.sh                 # forklift starts at (-7.5, 2.5) m, forks pointing down
#   ./run_fake_pick_test.sh 8.0 3.0 180     # start x y [m] and body yaw [deg] (forks = yaw + 180)
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
set +u; source /opt/ros/jazzy/setup.bash; set -u
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID_TEST:-81}
python3 tests/fake_world.py "$@" > tests/fake_world.log 2>&1 & W=$!
python3 rl_forklift_controller.py 1 > tests/rl_node.log 2>&1 & N=$!
trap 'kill $N $W 2>/dev/null || true' EXIT
sleep 2
timeout 400 python3 tests/pick_place_cycle.py 2>&1 | grep -E "^>|status|TIMEOUT|^  "
echo "logs: tests/rl_node.log, tests/fake_world.log (TRUE ... tips in pallet frame = ground truth)"
