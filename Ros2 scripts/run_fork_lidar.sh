#!/usr/bin/env bash
# Prints how to attach the 3x64 fork lidar (/forklift_N/scan, needed by the insert policy).
# The lidar has to run inside the Isaac Sim session that has Environment.usd open, so this script cannot
# start it from a terminal. It does not edit any USD.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIDAR="${SCRIPT_DIR}/attach_fork_lidar.py"

echo "Fork lidar runs inside Isaac Sim:"
echo "  1. Start Isaac Sim from a terminal where ROS 2 is sourced (source /opt/ros/jazzy/setup.bash),"
echo "     open Environment.usd and press Play"
echo "  2. Window -> Script Editor -> File -> Open:"
echo "       ${LIDAR}"
echo "  3. Run (Ctrl+Enter). It hooks the physics step, so the sim keeps running; running it again"
echo "     replaces the old one. Scans only publish while the scene is playing."
echo "Check:  ros2 topic hz /forklift_1/scan      (about 10 Hz, 192 ranges)"
