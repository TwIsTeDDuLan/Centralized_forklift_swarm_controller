#!/usr/bin/env bash
set -eo pipefail

cd "$(dirname "$0")"

# Load ROS 2 environment for rclpy/sensor_msgs/cv_bridge.
source /opt/ros/jazzy/setup.bash
set -u

# Optional safe mode:
#   SWARM_DISABLE_ARUCO=1 ./run_tracker.sh
exec /usr/bin/python3 vision_tracker.py "$@"
